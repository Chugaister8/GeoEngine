/*
 * GeoEngine — Terrain Shader (WGSL)
 * WebGPU Shading Language для рендерингу терейну.
 *
 * Pipeline:
 *   Vertex: висоти → clipspace + UV + normal + world pos
 *   Fragment: PBR освітлення + texture splatting + fog
 *
 * Texture splatting:
 *   Змішування текстур за висотою та крутизною схилу:
 *   - Низини  (< snow_line):  grass
 *   - Схили   (slope > 30°): rock
 *   - Вершини (> snow_line):  snow
 *   - Вода    (< sea_level):  sand
 *
 * Uniforms:
 *   camera:   view/projection матриці, позиція
 *   terrain:  параметри терейну (висоти, LOD)
 *   lighting: сонце, атмосфера
 *   fog:      параметри туману
 */

// ================================================================
// UNIFORM STRUCTS
// ================================================================

struct CameraUniforms {
    view:        mat4x4<f32>,
    projection:  mat4x4<f32>,
    view_proj:   mat4x4<f32>,
    position:    vec3<f32>,
    near:        f32,
    far:         f32,
    _pad:        vec3<f32>,
}

struct TerrainUniforms {
    // Висоти
    min_elevation:   f32,
    max_elevation:   f32,
    // LOD
    lod_level:       f32,
    morph_factor:    f32,   // 0=поточний LOD, 1=наступний LOD
    // Масштаб
    scale:           f32,
    uv_scale:        f32,
    // Texture splatting пороги
    sea_level:       f32,
    grass_end:       f32,
    rock_start:      f32,
    snow_start:      f32,
    slope_rock_deg:  f32,   // кут схилу з якого починається rock
    slope_snow_deg:  f32,
    _pad:            vec2<f32>,
}

struct LightingUniforms {
    sun_direction:  vec3<f32>,
    sun_intensity:  f32,
    sun_color:      vec3<f32>,
    ambient:        f32,
    shadow_softness: f32,
    _pad:           vec3<f32>,
}

struct FogUniforms {
    color:     vec3<f32>,
    density:   f32,
    start:     f32,
    end:       f32,
    height:    f32,   // висота де туман зникає
    _pad:      f32,
}

// ================================================================
// BINDINGS
// ================================================================

@group(0) @binding(0) var<uniform> camera:   CameraUniforms;
@group(0) @binding(1) var<uniform> terrain:  TerrainUniforms;
@group(0) @binding(2) var<uniform> lighting: LightingUniforms;
@group(0) @binding(3) var<uniform> fog:      FogUniforms;

// Текстури
@group(1) @binding(0) var terrain_sampler: sampler;
@group(1) @binding(1) var grass_tex:       texture_2d<f32>;
@group(1) @binding(2) var rock_tex:        texture_2d<f32>;
@group(1) @binding(3) var snow_tex:        texture_2d<f32>;
@group(1) @binding(4) var sand_tex:        texture_2d<f32>;
@group(1) @binding(5) var normal_map:      texture_2d<f32>;
@group(1) @binding(6) var satellite_tex:   texture_2d<f32>;   // опційна
@group(1) @binding(7) var shadow_map:      texture_depth_2d;
@group(1) @binding(8) var shadow_sampler:  sampler_comparison;

// ================================================================
// VERTEX I/O
// ================================================================

struct VertexInput {
    @location(0) position: vec3<f32>,   // XYZ у ENU метрах
    @location(1) uv:       vec2<f32>,   // [0..1] текстурні координати
    @location(2) normal:   vec3<f32>,   // нормаль поверхні
}

struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       world_pos:   vec3<f32>,
    @location(1)       uv:          vec2<f32>,
    @location(2)       normal:      vec3<f32>,
    @location(3)       elevation:   f32,          // висота для splatting
    @location(4)       slope:       f32,          // крутизна для splatting
    @location(5)       fog_factor:  f32,          // [0..1] туман
    @location(6)       shadow_pos:  vec4<f32>,    // позиція у shadow map space
    @location(7)       view_dist:   f32,          // відстань від камери
}

// ================================================================
// VERTEX SHADER
// ================================================================

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // World position (ENU → world)
    let world_pos = vec4<f32>(in.position, 1.0);

    // Clip space
    out.clip_pos  = camera.view_proj * world_pos;
    out.world_pos = in.position;

    // UV масштаб залежно від LOD (ближче = більше деталей)
    out.uv = in.uv * terrain.uv_scale;

    // Нормаль (без трансформації для terrain — вже у world space)
    out.normal = normalize(in.normal);

    // Висота для texture splatting
    out.elevation = in.position.y;

    // Крутизна (кут між нормаллю та вертикаллю)
    let up       = vec3<f32>(0.0, 1.0, 0.0);
    let dot_up   = clamp(dot(out.normal, up), 0.0, 1.0);
    out.slope    = degrees(acos(dot_up));

    // Відстань від камери (для LOD morph та fog)
    let to_cam    = camera.position - in.position;
    out.view_dist = length(to_cam);

    // Fog factor (exponential squared)
    let fog_dist    = max(0.0, out.view_dist - fog.start);
    out.fog_factor  = 1.0 - exp(-fog.density * fog_dist * fog_dist);
    out.fog_factor  = clamp(out.fog_factor, 0.0, 1.0);

    // Висотний туман (зменшуємо на висоті)
    let height_fog  = clamp(1.0 - in.position.y / fog.height, 0.0, 1.0);
    out.fog_factor *= height_fog;

    // Shadow map position (TODO: shadow matrix uniform)
    out.shadow_pos  = world_pos;  // буде множитися на shadow_matrix

    return out;
}

// ================================================================
// FRAGMENT HELPERS
// ================================================================

// Smoothstep blend між двома текстурами з overlap зоною
fn blend_textures(t1: vec4<f32>, t2: vec4<f32>, factor: f32, overlap: f32) -> vec4<f32> {
    let blend = smoothstep(0.5 - overlap, 0.5 + overlap, factor);
    return mix(t1, t2, blend);
}

// PBR: Lambert diffuse
fn diffuse(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    return max(0.0, dot(normalize(normal), normalize(light_dir)));
}

// PBR: Blinn-Phong specular
fn specular(
    normal:    vec3<f32>,
    light_dir: vec3<f32>,
    view_dir:  vec3<f32>,
    shininess: f32,
) -> f32 {
    let half_dir = normalize(light_dir + view_dir);
    return pow(max(0.0, dot(normal, half_dir)), shininess);
}

// Fake ambient occlusion з нормалі (дешевий AO)
fn fake_ao(normal: vec3<f32>, elevation: f32, min_elev: f32) -> f32 {
    let height_factor = clamp((elevation - min_elev) / 100.0, 0.0, 1.0);
    let up_dot        = max(0.0, dot(normal, vec3<f32>(0.0, 1.0, 0.0)));
    return mix(0.7, 1.0, up_dot * height_factor);
}

// Texture splatting: змішати текстури за висотою та крутизною
fn splat_textures(
    uv:        vec2<f32>,
    elevation: f32,
    slope:     f32,
    sea_level: f32,
    grass_end: f32,
    rock_start: f32,
    snow_start: f32,
    slope_rock: f32,
    slope_snow: f32,
) -> vec4<f32> {
    // Базові текстури
    let grass_color  = textureSample(grass_tex,    terrain_sampler, uv);
    let rock_color   = textureSample(rock_tex,     terrain_sampler, uv);
    let snow_color   = textureSample(snow_tex,     terrain_sampler, uv);
    let sand_color   = textureSample(sand_tex,     terrain_sampler, uv);

    // Спочатку визначаємо базовий колір за висотою
    var base_color = grass_color;

    // Пісок/вода (нижче рівня моря)
    if elevation < sea_level {
        let water_blend = clamp((sea_level - elevation) / 5.0, 0.0, 1.0);
        base_color = mix(grass_color, sand_color, water_blend);
    }

    // Трава → скеля (за висотою)
    let height_rock = smoothstep(rock_start - 50.0, rock_start + 50.0, elevation);
    base_color = mix(base_color, rock_color, height_rock * 0.7);

    // Трава → скеля (за крутизною схилу)
    let slope_rock_factor = smoothstep(slope_rock - 10.0, slope_rock + 10.0, slope);
    base_color = mix(base_color, rock_color, slope_rock_factor);

    // Скеля → сніг (за висотою)
    let snow_factor = smoothstep(snow_start - 100.0, snow_start + 50.0, elevation);
    base_color = mix(base_color, snow_color, snow_factor);

    // На крутих схилах сніг не лежить (slope > 50°)
    let snow_slope_mask = smoothstep(slope_snow - 5.0, slope_snow + 10.0, slope);
    base_color = mix(base_color, rock_color, snow_slope_mask * snow_factor);

    return base_color;
}

// Normal mapping: трансформувати normal map у world space
fn apply_normal_map(
    uv:           vec2<f32>,
    surface_norm: vec3<f32>,
) -> vec3<f32> {
    let nm_sample = textureSample(normal_map, terrain_sampler, uv).xyz;
    // Розпакувати [0..1] → [-1..1]
    let tangent_normal = nm_sample * 2.0 - 1.0;

    // TBN матриця (спрощена для горизонтального терейну)
    let up      = vec3<f32>(0.0, 1.0, 0.0);
    let tangent = normalize(cross(surface_norm, vec3<f32>(0.0, 0.0, 1.0)));
    let bitang  = normalize(cross(surface_norm, tangent));

    let world_normal = tangent_normal.x * tangent
                     + tangent_normal.y * bitang
                     + tangent_normal.z * surface_norm;

    return normalize(mix(surface_norm, world_normal, 0.5));
}

// ================================================================
// FRAGMENT SHADER
// ================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    // ---- Normal mapping ----
    let uv_detail  = in.uv * 8.0;   // деталізовані UV для normal map
    let normal     = apply_normal_map(uv_detail, in.normal);

    // ---- Texture splatting ----
    var albedo = splat_textures(
        in.uv,
        in.elevation,
        in.slope,
        terrain.sea_level,
        terrain.grass_end,
        terrain.rock_start,
        terrain.snow_start,
        terrain.slope_rock_deg,
        terrain.slope_snow_deg,
    );

    // ---- Satellite overlay (якщо є) ----
    // Змішуємо satellite texture з procedural splatting
    let satellite   = textureSample(satellite_tex, terrain_sampler, in.uv);
    let sat_alpha   = satellite.a;   // 0 = немає даних, 1 = повне покриття
    albedo = mix(albedo, satellite, sat_alpha * 0.85);

    // ---- PBR Lighting ----
    let view_dir    = normalize(camera.position - in.world_pos);
    let light_dir   = normalize(lighting.sun_direction);

    // Diffuse (Lambert)
    let diff_factor = diffuse(normal, light_dir);
    let diffuse_col = lighting.sun_color * lighting.sun_intensity * diff_factor;

    // Specular (Blinn-Phong, слабкий для терейну)
    let spec_factor = specular(normal, light_dir, view_dir, 16.0) * 0.05;
    let spec_col    = lighting.sun_color * spec_factor;

    // Ambient
    let ambient_col = lighting.sun_color * lighting.ambient;

    // AO
    let ao = fake_ao(normal, in.elevation, terrain.min_elevation);

    // Фінальне освітлення
    var lit_color = albedo.rgb * (diffuse_col + ambient_col) * ao + spec_col;

    // ---- Distance fade для деталей ----
    // На відстані ховаємо текстурний шум (триплінарний)
    let dist_fade = clamp(in.view_dist / 5000.0, 0.0, 1.0);
    let avg_color = (lit_color.r + lit_color.g + lit_color.b) / 3.0;
    lit_color     = mix(lit_color, vec3<f32>(avg_color), dist_fade * 0.3);

    // ---- Elevation-based color grading ----
    // Тепліші тони в долинах, холодніші на вершинах
    let norm_elev   = clamp(
        (in.elevation - terrain.min_elevation)
        / max(1.0, terrain.max_elevation - terrain.min_elevation),
        0.0, 1.0,
    );
    let warm_shift  = vec3<f32>(0.05, 0.02, -0.03) * (1.0 - norm_elev);
    let cold_shift  = vec3<f32>(-0.02, -0.01, 0.05) * norm_elev;
    lit_color      += warm_shift + cold_shift;

    // ---- Fog ----
    lit_color = mix(lit_color, fog.color, in.fog_factor);

    // ---- Gamma correction (linear → sRGB) ----
    let gamma_corrected = pow(max(vec3<f32>(0.0), lit_color), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, albedo.a);
}
