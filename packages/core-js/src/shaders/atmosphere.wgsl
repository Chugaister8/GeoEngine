/*
 * GeoEngine — Atmosphere Shader (WGSL)
 * Фізично коректне атмосферне розсіювання.
 *
 * Реалізує:
 *   - Rayleigh scattering (молекули) → блакитне небо
 *   - Mie scattering (аерозолі/пил) → серпанок, сонячний диск
 *   - Sunset/sunrise кольори
 *   - Зіркове небо (вночі)
 *   - Місяць (placeholder)
 *
 * Алгоритм: Nishita (1993) + Preetham sky model.
 * Render: fullscreen quad (або skybox cube).
 *
 * Uniforms:
 *   camera:     позиція та напрямок
 *   atmosphere: параметри атмосфери
 *   time:       час доби + дата для позиції сонця
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

struct AtmosphereUniforms {
    // Параметри планети
    planet_radius:  f32,    // метрів (6371000 для Землі)
    atmos_radius:   f32,    // планета + атмосфера (6471000)
    // Rayleigh
    rayleigh_coeff: vec3<f32>,   // коефіцієнти для RGB
    rayleigh_scale: f32,         // scale height (8500м)
    // Mie
    mie_coeff:      f32,
    mie_scale:      f32,         // scale height (1200м)
    mie_dir:        f32,         // directional (Henyey-Greenstein g)
    // Інтенсивність
    sun_intensity:  f32,
    // Зірки
    star_threshold: f32,    // яскравість вище якої видно зірки
    star_brightness: f32,
    _pad:           vec2<f32>,
}

struct TimeUniforms {
    sun_direction:  vec3<f32>,   // нормалізований вектор до сонця
    time_of_day:    f32,         // [0..1] де 0=північ, 0.5=полудень
    moon_direction: vec3<f32>,
    moon_phase:     f32,         // [0..1] фаза місяця
}

@group(0) @binding(0) var<uniform> camera:     CameraUniforms;
@group(0) @binding(1) var<uniform> atmosphere: AtmosphereUniforms;
@group(0) @binding(2) var<uniform> time_uni:   TimeUniforms;

@group(1) @binding(0) var sky_sampler: sampler;
@group(1) @binding(1) var star_tex:    texture_2d<f32>;   // зоряна карта

// ================================================================
// VERTEX SHADER (fullscreen triangle)
// ================================================================

struct SkyVertexOutput {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       ray_dir:    vec3<f32>,   // напрямок погляду
    @location(1)       uv:         vec2<f32>,
}

// Fullscreen triangle trick (три вершини = повний екран)
@vertex
fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyVertexOutput {
    var out: SkyVertexOutput;

    // Дві великих трикутники що покривають екран
    let uv = vec2<f32>(
        f32((idx << 1u) & 2u),
        f32(idx & 2u),
    );
    out.uv = uv;

    let pos = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);
    out.clip_pos = pos;

    // Ray direction з inverse view-projection
    let inv_proj = /* inverse(camera.projection) */ mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    let view_ray = inv_proj * vec4<f32>(pos.xy, 1.0, 1.0);
    out.ray_dir  = (transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    )) * view_ray.xyz);

    return out;
}

// ================================================================
// RAYLEIGH / MIE FUNCTIONS
// ================================================================

// Rayleigh phase function
fn rayleigh_phase(cos_theta: f32) -> f32 {
    let factor = 3.0 / (16.0 * 3.14159265);
    return factor * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein Mie phase function
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2  = g * g;
    let num = (1.0 - g2);
    let den = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (3.0 / (8.0 * 3.14159265)) * (num / den) * ((1.0 + cos_theta * cos_theta) / (2.0 + g2));
}

// Перетин ray зі сферою
// Повертає (t_near, t_far), де t < 0 = немає перетину
fn ray_sphere_intersect(
    ray_origin: vec3<f32>,
    ray_dir:    vec3<f32>,
    radius:     f32,
) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - radius * radius;
    let d = b * b - c;

    if d < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }

    let sqrt_d = sqrt(d);
    return vec2<f32>(-b - sqrt_d, -b + sqrt_d);
}

// Density функції
fn rayleigh_density(height: f32) -> f32 {
    return exp(-height / atmosphere.rayleigh_scale);
}

fn mie_density(height: f32) -> f32 {
    return exp(-height / atmosphere.mie_scale);
}

// ================================================================
// ГОЛОВНА АТМОСФЕРНА ІНТЕГРАЦІЯ
// ================================================================

const NUM_SAMPLES:       i32 = 16;
const NUM_LIGHT_SAMPLES: i32 = 8;

fn compute_atmosphere(
    ray_origin: vec3<f32>,   // позиція спостерігача (на поверхні планети)
    ray_dir:    vec3<f32>,   // напрямок погляду
    sun_dir:    vec3<f32>,   // напрямок до сонця
) -> vec3<f32> {

    let planet_r = atmosphere.planet_radius;
    let atmos_r  = atmosphere.atmos_radius;

    // Позиція відносно центру планети
    let origin = ray_origin + vec3<f32>(0.0, planet_r, 0.0);

    // Перетин з атмосферою
    let atmos_hit = ray_sphere_intersect(origin, ray_dir, atmos_r);
    if atmos_hit.y < 0.0 {
        return vec3<f32>(0.0);  // промінь не перетинає атмосферу
    }

    let t_start = max(0.0, atmos_hit.x);
    let t_end   = atmos_hit.y;

    if t_start >= t_end {
        return vec3<f32>(0.0);
    }

    // Кут між сонцем та напрямком погляду
    let cos_theta = dot(ray_dir, sun_dir);

    // Phase functions
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, atmosphere.mie_dir);

    // Інтегрування вздовж ray
    let step_size = (t_end - t_start) / f32(NUM_SAMPLES);
    var t         = t_start + step_size * 0.5;

    var rayleigh_sum = vec3<f32>(0.0);
    var mie_sum      = vec3<f32>(0.0);
    var opt_depth_r  = 0.0;
    var opt_depth_m  = 0.0;

    for (var i = 0; i < NUM_SAMPLES; i++) {
        let pos    = origin + ray_dir * t;
        let height = max(0.0, length(pos) - planet_r);

        let density_r = rayleigh_density(height) * step_size;
        let density_m = mie_density(height) * step_size;

        opt_depth_r += density_r;
        opt_depth_m += density_m;

        // Інтеграція до сонця (light march)
        let sun_hit = ray_sphere_intersect(pos, sun_dir, atmos_r);
        let step_l  = sun_hit.y / f32(NUM_LIGHT_SAMPLES);
        var tl      = step_l * 0.5;
        var opt_r_l = 0.0;
        var opt_m_l = 0.0;

        for (var j = 0; j < NUM_LIGHT_SAMPLES; j++) {
            let pos_l    = pos + sun_dir * tl;
            let height_l = max(0.0, length(pos_l) - planet_r);
            opt_r_l += rayleigh_density(height_l) * step_l;
            opt_m_l += mie_density(height_l) * step_l;
            tl += step_l;
        }

        // Transmittance
        let transmit = exp(
            -(atmosphere.rayleigh_coeff * (opt_depth_r + opt_r_l))
            - atmosphere.mie_coeff * 1.1 * (opt_depth_m + opt_m_l)
        );

        rayleigh_sum += density_r * transmit;
        mie_sum      += density_m * transmit;
        t            += step_size;
    }

    let sun_i = atmosphere.sun_intensity;
    return sun_i * (
        rayleigh_sum * atmosphere.rayleigh_coeff * phase_r
        + mie_sum * atmosphere.mie_coeff * phase_m
    );
}

// ================================================================
// ЗІРКИ
// ================================================================

fn star_field(ray_dir: vec3<f32>, time_of_day: f32) -> vec3<f32> {
    // Зірки видно тільки вночі
    let night_factor = clamp(1.0 - time_of_day * 2.0, 0.0, 1.0)
                     + clamp((time_of_day - 0.5) * 2.0, 0.0, 1.0);

    if night_factor < 0.01 {
        return vec3<f32>(0.0);
    }

    // Spherical UV для зоряної текстури
    let uv = vec2<f32>(
        atan2(ray_dir.z, ray_dir.x) / (2.0 * 3.14159265) + 0.5,
        asin(clamp(ray_dir.y, -1.0, 1.0)) / 3.14159265 + 0.5,
    );

    let stars = textureSample(star_tex, sky_sampler, uv);
    let bright = step(atmosphere.star_threshold, stars.r);

    return stars.rgb * bright * atmosphere.star_brightness * night_factor;
}

// ================================================================
// SUN DISK
// ================================================================

fn sun_disk(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cos_angle = dot(ray_dir, sun_dir);
    let sun_size  = 0.9998;   // cos(0.54° halfangle) ≈ кутовий розмір Сонця

    if cos_angle < sun_size {
        return vec3<f32>(0.0);
    }

    let edge    = smoothstep(sun_size, sun_size + 0.0001, cos_angle);
    let limb    = 1.0 - smoothstep(0.9999, 1.0, cos_angle) * 0.3;  // limb darkening
    return vec3<f32>(1.0, 0.95, 0.85) * edge * limb * atmosphere.sun_intensity * 3.0;
}

// ================================================================
// FRAGMENT SHADER
// ================================================================

@fragment
fn fs_sky(in: SkyVertexOutput) -> @location(0) vec4<f32> {
    let ray_dir  = normalize(in.ray_dir);
    let sun_dir  = normalize(time_uni.sun_direction);

    // Не рендеримо атмосферу нижче горизонту
    if ray_dir.y < -0.05 {
        // Земля (темно-коричневий)
        let ground = vec3<f32>(0.12, 0.10, 0.08);
        return vec4<f32>(ground, 1.0);
    }

    // Позиція спостерігача (1.8м над рівнем моря)
    let observer_pos = vec3<f32>(camera.position.x, camera.position.y + 1.8, camera.position.z);

    // Атмосферне розсіювання
    var sky_color = compute_atmosphere(observer_pos, ray_dir, sun_dir);

    // Сонячний диск
    sky_color += sun_disk(ray_dir, sun_dir);

    // Зірки
    sky_color += star_field(ray_dir, time_uni.time_of_day);

    // HDR tone mapping (Reinhard)
    sky_color = sky_color / (sky_color + vec3<f32>(1.0));

    // Gamma correction
    sky_color = pow(max(vec3<f32>(0.0), sky_color), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(sky_color, 1.0);
}
