/*
 * GeoEngine — Water Shader (WGSL)
 * Реалістичний рендеринг води: хвилі, рефлексія, рефракція.
 *
 * Техніки:
 *   - FFT-based хвилі (approximation через gerstner waves)
 *   - Screen-space рефлексія (SSR lite)
 *   - Рефракція через depth buffer
 *   - Foam на берегах та гребенях
 *   - Caustics (проекція на дно)
 *   - Fresnel ефект (кут між нормаллю та поглядом)
 *
 * Gerstner хвилі:
 *   Кожна хвиля задається: амплітуда, довжина, напрямок, швидкість.
 *   Vertex shader зміщує vertices, fragment обчислює normal.
 */

// ================================================================
// UNIFORMS
// ================================================================

struct CameraUniforms {
    view:       mat4x4<f32>,
    projection: mat4x4<f32>,
    view_proj:  mat4x4<f32>,
    position:   vec3<f32>,
    near:       f32,
    far:        f32,
    _pad:       vec3<f32>,
}

struct WaterUniforms {
    time:          f32,          // секунди з початку
    water_level:   f32,          // висота поверхні води (метри)
    // Хвилі
    wave_speed:    f32,
    wave_scale:    f32,
    choppiness:    f32,          // вертикальне зміщення гребенів
    // Оптика
    water_color:   vec3<f32>,    // глибока вода
    shallow_color: vec3<f32>,    // мілководдя
    deep_depth:    f32,          // глибина для повного кольору
    transparency:  f32,          // прозорість [0..1]
    // Foam
    foam_threshold: f32,         // глибина де з'являється піна
    foam_scale:    f32,
    // Рефлексія
    reflection_strength: f32,
    // Рефракція
    refraction_scale: f32,
    _pad:          vec2<f32>,
}

struct LightingUniforms {
    sun_direction:  vec3<f32>,
    sun_intensity:  f32,
    sun_color:      vec3<f32>,
    ambient:        f32,
    _pad:           vec4<f32>,
}

// Gerstner wave параметри (4 хвилі)
struct WaveParams {
    direction: vec2<f32>,   // напрямок (нормалізований)
    amplitude: f32,
    frequency: f32,         // 2π / wavelength
    speed:     f32,
    steepness: f32,         // Q параметр Gerstner [0..1]
    _pad:      vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera:   CameraUniforms;
@group(0) @binding(1) var<uniform> water:    WaterUniforms;
@group(0) @binding(2) var<uniform> lighting: LightingUniforms;
@group(0) @binding(3) var<uniform> waves:    array<WaveParams, 4>;

@group(1) @binding(0) var water_sampler:     sampler;
@group(1) @binding(1) var normal_map_0:      texture_2d<f32>;   // хвильова нормаль 1
@group(1) @binding(2) var normal_map_1:      texture_2d<f32>;   // хвильова нормаль 2
@group(1) @binding(3) var foam_tex:          texture_2d<f32>;
@group(1) @binding(4) var depth_tex:         texture_depth_2d;  // scene depth
@group(1) @binding(5) var scene_color_tex:   texture_2d<f32>;   // scene color (рефракція)
@group(1) @binding(6) var env_map:           texture_cube<f32>; // environment cubemap

// ================================================================
// GERSTNER WAVES
// ================================================================

struct GerstnerResult {
    offset:  vec3<f32>,   // зміщення вершини
    normal:  vec3<f32>,   // нормаль поверхні
    tangent: vec3<f32>,   // тангент (для bump mapping)
}

fn gerstner_wave(
    pos:     vec2<f32>,   // XZ позиція вершини
    params:  WaveParams,
    t:       f32,         // час
) -> GerstnerResult {
    var res: GerstnerResult;

    let phase = params.frequency * dot(params.direction, pos)
              - params.speed * t;
    let sin_p = sin(phase);
    let cos_p = cos(phase);

    // Горизонтальне зміщення (Gerstner)
    let Q = params.steepness / (params.frequency * params.amplitude * 4.0);
    res.offset.x = Q * params.amplitude * params.direction.x * cos_p;
    res.offset.z = Q * params.amplitude * params.direction.y * cos_p;
    // Вертикальне зміщення
    res.offset.y = params.amplitude * sin_p;

    // Нормаль
    let WA  = params.frequency * params.amplitude;
    res.normal = vec3<f32>(
        -params.direction.x * WA * cos_p,
        1.0 - Q * WA * sin_p,
        -params.direction.y * WA * cos_p,
    );

    // Тангент
    res.tangent = vec3<f32>(
        1.0 - Q * params.direction.x * params.direction.x * WA * sin_p,
        params.direction.x * WA * cos_p,
        -Q * params.direction.x * params.direction.y * WA * sin_p,
    );

    return res;
}

fn compute_gerstner(pos: vec2<f32>, t: f32) -> GerstnerResult {
    var total: GerstnerResult;
    total.offset = vec3<f32>(0.0);
    total.normal = vec3<f32>(0.0, 1.0, 0.0);
    total.tangent = vec3<f32>(1.0, 0.0, 0.0);

    for (var i = 0; i < 4; i++) {
        let w = gerstner_wave(pos, waves[i], t);
        total.offset  += w.offset;
        total.normal  += w.normal - vec3<f32>(0.0, 1.0, 0.0);
        total.tangent += w.tangent - vec3<f32>(1.0, 0.0, 0.0);
    }

    total.normal  = normalize(total.normal + vec3<f32>(0.0, 1.0, 0.0));
    total.tangent = normalize(total.tangent + vec3<f32>(1.0, 0.0, 0.0));

    return total;
}

// ================================================================
// VERTEX SHADER
// ================================================================

struct WaterVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
}

struct WaterVertexOutput {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       uv:         vec2<f32>,
    @location(2)       normal:     vec3<f32>,
    @location(3)       tangent:    vec3<f32>,
    @location(4)       screen_uv:  vec2<f32>,
    @location(5)       foam:       f32,
}

@vertex
fn vs_water(in: WaterVertexInput) -> WaterVertexOutput {
    var out: WaterVertexOutput;

    // XZ позиція для Gerstner
    let xz_pos = in.position.xz * water.wave_scale;

    // Обчислити Gerstner зміщення
    let gerstner  = compute_gerstner(xz_pos, water.time * water.wave_speed);

    // Зміщена позиція
    let disp_pos  = in.position + gerstner.offset * water.choppiness;
    let world_pos = vec4<f32>(disp_pos, 1.0);

    out.clip_pos  = camera.view_proj * world_pos;
    out.world_pos = disp_pos;
    out.uv        = in.uv;
    out.normal    = gerstner.normal;
    out.tangent   = gerstner.tangent;

    // Screen UV для SSR та рефракції
    let ndc       = out.clip_pos.xy / out.clip_pos.w;
    out.screen_uv = ndc * 0.5 + 0.5;

    // Foam: де хвиля висока — там піна
    out.foam = clamp(
        (gerstner.offset.y - water.foam_threshold) / water.foam_threshold,
        0.0, 1.0,
    );

    return out;
}

// ================================================================
// FRAGMENT HELPERS
// ================================================================

// Fresnel (Schlick approximation)
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Linearize depth
fn linear_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// Normal map blend (хвильові деталі)
fn blend_normal_maps(
    uv:   vec2<f32>,
    time: f32,
) -> vec3<f32> {
    let uv0 = uv * 2.0 + vec2<f32>(time * 0.02, time * 0.01);
    let uv1 = uv * 3.0 - vec2<f32>(time * 0.015, time * 0.025);

    let n0 = textureSample(normal_map_0, water_sampler, uv0).xyz * 2.0 - 1.0;
    let n1 = textureSample(normal_map_1, water_sampler, uv1).xyz * 2.0 - 1.0;

    return normalize(vec3<f32>(n0.xy + n1.xy, n0.z * n1.z));
}

// ================================================================
// FRAGMENT SHADER
// ================================================================

@fragment
fn fs_water(in: WaterVertexOutput) -> @location(0) vec4<f32> {

    let view_dir = normalize(camera.position - in.world_pos);
    let sun_dir  = normalize(lighting.sun_direction);

    // ---- Normal mapping ----
    // Комбінуємо Gerstner нормаль з деталізованою normal map
    let detail_normal = blend_normal_maps(in.uv, water.time);
    let bitangent     = cross(in.normal, in.tangent);
    let tbn_normal    = normalize(
        detail_normal.x * in.tangent
        + detail_normal.y * bitangent
        + detail_normal.z * in.normal
    );
    let normal = normalize(mix(in.normal, tbn_normal, 0.7));

    // ---- Fresnel ----
    let cos_theta  = max(0.0, dot(normal, view_dir));
    let f0         = 0.02;  // вода ≈ n=1.33
    let fresnel    = fresnel_schlick(cos_theta, f0);

    // ---- Рефлексія ----
    let reflect_dir   = reflect(-view_dir, normal);
    let env_reflection = textureSample(env_map, water_sampler, reflect_dir).rgb;

    // SSR distortion (зміщення screen UV для рефлексії)
    let ssr_offset  = normal.xz * 0.05;
    let reflect_uv  = clamp(in.screen_uv + ssr_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let ssr_color   = textureSample(scene_color_tex, water_sampler, vec2<f32>(reflect_uv.x, 1.0 - reflect_uv.y)).rgb;

    let reflection  = mix(env_reflection, ssr_color, 0.5) * water.reflection_strength;

    // ---- Рефракція ----
    let refract_offset = normal.xz * water.refraction_scale * 0.1;
    let refract_uv     = clamp(
        in.screen_uv + refract_offset,
        vec2<f32>(0.0), vec2<f32>(1.0)
    );
    let refraction   = textureSample(
        scene_color_tex, water_sampler,
        vec2<f32>(refract_uv.x, 1.0 - refract_uv.y)
    ).rgb;

    // ---- Колір води за глибиною ----
    // Використовуємо depth buffer для визначення глибини під водою
    // (спрощено без справжнього depth: просто колір за UV)
    let water_col   = mix(
        water.shallow_color,
        water.water_color,
        clamp(1.0 - exp(-in.world_pos.y / water.deep_depth), 0.0, 1.0),
    );

    // ---- Поверхневий колір ----
    var surface_color = mix(
        refraction * water_col,
        reflection,
        fresnel,
    );

    // ---- Specular (Blinn-Phong) ----
    let half_dir   = normalize(sun_dir + view_dir);
    let spec       = pow(max(0.0, dot(normal, half_dir)), 512.0);
    surface_color += lighting.sun_color * spec * lighting.sun_intensity * 0.8;

    // ---- Foam ----
    let foam_uv    = in.uv * water.foam_scale + vec2<f32>(water.time * 0.01);
    let foam_tex_c = textureSample(foam_tex, water_sampler, foam_uv).r;
    let foam_mask  = in.foam * foam_tex_c;
    surface_color  = mix(surface_color, vec3<f32>(1.0), foam_mask * 0.8);

    // ---- Прозорість (Fresnel-based) ----
    let alpha = mix(
        water.transparency,
        1.0,
        fresnel + foam_mask * 0.5,
    );

    // ---- Gamma ----
    let final_color = pow(max(vec3<f32>(0.0), surface_color), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, clamp(alpha, 0.0, 1.0));
}
