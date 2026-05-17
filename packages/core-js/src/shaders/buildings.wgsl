/*
 * GeoEngine — Buildings Shader (WGSL)
 * Рендеринг 3D будівель з OSM даних.
 *
 * Особливості:
 *   - Instanced rendering (тисячі будівель за один draw call)
 *   - Per-instance колір (тип будівлі)
 *   - Процедурні вікна (UV-based)
 *   - Нічне підсвічування (emissive вікна)
 *   - LOD: далеко = billboard, близько = повна геометрія
 *   - Edge highlighting для виділення будівлі
 *
 * Instance data (per-building):
 *   - Transform матриця (позиція + поворот)
 *   - Колір (RGBA)
 *   - Висота
 *   - OSM ID (для picking)
 */

// ================================================================
// UNIFORMS
// ================================================================

struct CameraUniforms {
    view:         mat4x4<f32>,
    projection:   mat4x4<f32>,
    view_proj:    mat4x4<f32>,
    position:     vec3<f32>,
    near:         f32,
    far:          f32,
    _pad:         vec3<f32>,
}

struct BuildingUniforms {
    time:             f32,
    window_rows:      f32,      // кількість рядів вікон на поверх
    window_cols:      f32,      // вікон по горизонталі на 10м
    window_light_prob: f32,     // [0..1] ймовірність що вікно освітлене
    // LOD
    lod_distance:     f32,      // відстань переключення LOD
    // Highlighting
    highlighted_id:   u32,      // OSM ID виділеної будівлі
    highlight_color:  vec3<f32>,
    edge_width:       f32,
    // Night mode
    night_factor:     f32,      // [0..1] наскільки темно
    _pad:             vec3<f32>,
}

struct LightingUniforms {
    sun_direction:  vec3<f32>,
    sun_intensity:  f32,
    sun_color:      vec3<f32>,
    ambient:        f32,
    _pad:           vec4<f32>,
}

// Instance data (per building)
struct InstanceData {
    transform:  mat4x4<f32>,
    color:      vec4<f32>,
    height:     f32,
    osm_id:     u32,
    _pad:       vec2<f32>,
}

@group(0) @binding(0) var<uniform>       camera:    CameraUniforms;
@group(0) @binding(1) var<uniform>       buildings: BuildingUniforms;
@group(0) @binding(2) var<uniform>       lighting:  LightingUniforms;
@group(0) @binding(3) var<storage, read> instances: array<InstanceData>;

@group(1) @binding(0) var bld_sampler:   sampler;
@group(1) @binding(1) var facade_tex:    texture_2d<f32>;   // текстура фасаду
@group(1) @binding(2) var window_tex:    texture_2d<f32>;   // маска вікон
@group(1) @binding(3) var roof_tex:      texture_2d<f32>;   // текстура даху
@group(1) @binding(4) var noise_tex:     texture_2d<f32>;   // noise для варіацій

// ================================================================
// VERTEX SHADER
// ================================================================

struct BuildingVertexInput {
    @location(0) position:     vec3<f32>,
    @location(1) uv:           vec2<f32>,
    @location(2) normal:       vec3<f32>,
    @builtin(instance_index)   instance_idx: u32,
}

struct BuildingVertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       world_pos:   vec3<f32>,
    @location(1)       uv:          vec2<f32>,
    @location(2)       normal:      vec3<f32>,
    @location(3)       color:       vec4<f32>,
    @location(4)       height:      f32,
    @location(5)       is_roof:     f32,     // 0=стіна, 1=дах
    @location(6)       osm_id:      u32,
    @location(7)       view_dist:   f32,
}

@vertex
fn vs_building(in: BuildingVertexInput) -> BuildingVertexOutput {
    var out: BuildingVertexOutput;

    let inst = instances[in.instance_idx];

    // World position через instance transform
    let world_pos = inst.transform * vec4<f32>(in.position, 1.0);

    out.clip_pos  = camera.view_proj * world_pos;
    out.world_pos = world_pos.xyz;

    // Нормаль (трансформована без Translation)
    let normal_mat = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );
    out.normal = normalize(normal_mat * in.normal);

    out.uv        = in.uv;
    out.color     = inst.color;
    out.height    = inst.height;
    out.osm_id    = inst.osm_id;

    // Визначаємо дах по нормалі (дивиться вгору)
    out.is_roof = step(0.9, dot(out.normal, vec3<f32>(0.0, 1.0, 0.0)));

    // Відстань від камери
    out.view_dist = length(camera.position - world_pos.xyz);

    return out;
}

// ================================================================
// WINDOW GRID
// ================================================================

// Процедурні вікна — повертає 1.0 якщо піксель — вікно
fn window_mask(uv: vec2<f32>, height: f32, osm_id: u32) -> f32 {
    // Кількість вікон залежить від висоти будівлі
    let floors = max(1.0, height / 3.2);
    let rows   = floors * buildings.window_rows;
    let cols   = buildings.window_cols;

    // UV у просторі вікна
    let window_uv = fract(uv * vec2<f32>(cols, rows));

    // Вікно займає 60% клітинки, рама — 40%
    let frame = 0.15;
    let is_window = step(frame, window_uv.x) * step(window_uv.x, 1.0 - frame)
                  * step(frame, window_uv.y) * step(window_uv.y, 1.0 - frame);

    if is_window < 0.5 {
        return 0.0;
    }

    // Псевдорандомне освітлення вікна (детерміноване)
    let cell_x = floor(uv.x * cols);
    let cell_y = floor(uv.y * rows);
    let seed   = cell_x * 127.1 + cell_y * 311.7 + f32(osm_id) * 0.001;
    let rand   = fract(sin(seed) * 43758.5453);

    return step(1.0 - buildings.window_light_prob, rand);
}

// ================================================================
// FRAGMENT SHADER
// ================================================================

@fragment
fn fs_building(in: BuildingVertexOutput) -> @location(0) vec4<f32> {

    let view_dir  = normalize(camera.position - in.world_pos);
    let sun_dir   = normalize(lighting.sun_direction);
    let normal    = normalize(in.normal);

    // ---- Базовий колір ----
    var base_color: vec3<f32>;

    if in.is_roof > 0.5 {
        // Дах
        let roof_sample = textureSample(roof_tex, bld_sampler, in.uv * 4.0);
        base_color = in.color.rgb * roof_sample.rgb * 0.9;
    } else {
        // Фасад
        let facade_sample = textureSample(facade_tex, bld_sampler, in.uv * 2.0);
        base_color = in.color.rgb * facade_sample.rgb;

        // Noise для варіацій між будівлями (дрібні деталі)
        let noise_uv  = in.uv * 8.0 + vec2<f32>(f32(in.osm_id % 7u) * 0.1);
        let noise_val = textureSample(noise_tex, bld_sampler, noise_uv).r;
        base_color   *= mix(0.95, 1.05, noise_val);
    }

    // ---- Вікна (тільки для стін) ----
    var emissive = vec3<f32>(0.0);
    if in.is_roof < 0.5 {
        let win = window_mask(in.uv, in.height, in.osm_id);

        if win > 0.5 {
            // Колір освітленого вікна
            let warm_light = vec3<f32>(1.0, 0.85, 0.55);  // тепле жовте
            let cold_light = vec3<f32>(0.7, 0.85, 1.0);   // холодне синє

            // Мікс теплого/холодного за псевдорандомом
            let seed_c = f32(in.osm_id % 13u) * 0.1;
            let rand_c = fract(sin(seed_c) * 43758.5);
            let win_color = mix(warm_light, cold_light, rand_c);

            // Вікно емісивне вночі
            emissive = win_color * win * buildings.night_factor * 2.0;

            // Вдень вікна трохи відбивають небо
            base_color = mix(base_color, vec3<f32>(0.7, 0.85, 1.0), win * 0.3 * (1.0 - buildings.night_factor));
        }
    }

    // ---- PBR Lighting ----
    let diff_factor = max(0.0, dot(normal, sun_dir));
    let diffuse_col = lighting.sun_color * lighting.sun_intensity * diff_factor;

    // Specular (слабкий для будівель — вони не блищать)
    let half_dir   = normalize(sun_dir + view_dir);
    let spec_raw   = pow(max(0.0, dot(normal, half_dir)), 32.0);
    let spec_col   = lighting.sun_color * spec_raw * 0.05;

    // Ambient (враховує нічний режим)
    let ambient_day   = lighting.sun_color * lighting.ambient;
    let ambient_night = vec3<f32>(0.05, 0.06, 0.10);  // синюватий місячний
    let ambient_col   = mix(ambient_day, ambient_night, buildings.night_factor);

    // Фінальне денне освітлення
    var lit_color = base_color * (diffuse_col + ambient_col) + spec_col;

    // Нічний режим: зменшуємо денне освітлення
    lit_color = mix(lit_color, base_color * 0.1, buildings.night_factor * 0.8);

    // Додаємо emissive (вікна)
    lit_color += emissive;

    // ---- Highlighting ----
    if in.osm_id == buildings.highlighted_id {
        // Wireframe edge highlight (наближення через UV)
        let edge_x = min(in.uv.x, 1.0 - in.uv.x);
        let edge_y = min(in.uv.y, 1.0 - in.uv.y);
        let edge   = step(edge_x, buildings.edge_width) + step(edge_y, buildings.edge_width);
        lit_color  = mix(lit_color, buildings.highlight_color, clamp(edge, 0.0, 1.0) * 0.8);
    }

    // ---- LOD fade (billboarding on distance) ----
    let lod_fade = smoothstep(
        buildings.lod_distance * 0.9,
        buildings.lod_distance,
        in.view_dist,
    );
    // На великій відстані спрощуємо деталі
    lit_color = mix(lit_color, in.color.rgb * lighting.ambient, lod_fade * 0.5);

    // ---- Gamma correction ----
    let final_color = pow(max(vec3<f32>(0.0), lit_color), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, in.color.a);
}

// ================================================================
// PICKING PASS (окремий render pass для mouse picking)
// ================================================================

struct PickingOutput {
    @location(0) osm_id: vec4<u32>,
}

@fragment
fn fs_picking(in: BuildingVertexOutput) -> PickingOutput {
    // Кодуємо OSM ID у RGBA (32 bit)
    let id = in.osm_id;
    return PickingOutput(
        osm_id = vec4<u32>(
            (id >> 24u) & 0xFFu,
            (id >> 16u) & 0xFFu,
            (id >>  8u) & 0xFFu,
             id         & 0xFFu,
        )
    );
}
