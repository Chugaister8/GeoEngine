# ◈◈◈ GEOENGINE ◈◈◈


> *"Not just a 3D engine. A living, breathing digital twin of Earth —  
> scriptable in Python, rendered in WebGPU, powered by real data."*

---

## КОНЦЕПЦІЯ

GeoEngine — це **геопросторова платформа**,
що поєднує в собі:

- **3D рушій** з планетарним масштабом та LOD-рендерингом
- **GIS-аналітику** з реальними даними з відкритих і комерційних джерел  
- **AI/ML шар** для автоматичного розпізнавання та генерації контенту
- **Фізичний рушій** для симуляцій рельєфу, рідин і об'єктів
- **Python-first API** — повноцінна мова першого класу
- **WebGPU рендер** — технологія наступного десятиліття

---

## ТЕХНОЛОГІЧНЕ ЯДРО

| Шар | Технологія | Призначення |
|-----|-----------|-------------|
| **Рендеринг** | WebGPU / Three.js r165+ | Планетарний 3D рендер |
| **Фізика** | Rapier (Rust→WASM) | Тверді тіла, рідини, балістика |
| **Геопросторові дані** | rasterio, pyproj, GDAL | DEM, GeoTIFF, проекції |
| **3D Mesh** | trimesh, Open3D, PyVista | Геометрія, хмари точок |
| **AI / CV** | PyTorch, SAM2, SegFormer | Розпізнавання знімків |
| **Генерація** | Stable Diffusion, NeRF | AI контент |
| **Bridge** | FastAPI + WebSocket | Python ↔ JS реальний час |
| **Offline** | Pyodide / WASM | Браузер без сервера |
| **XR** | WebXR API | VR / AR / MR |
| **Big Data** | DuckDB, Apache Arrow, GeoParquet | Терабайтні геодані |

---

## СФЕРИ ЗАСТОСУВАННЯ

---

### ▸ 01 — ТОПОГРАФІЯ ТА ГЕОДЕЗІЯ

```
ЗАСТОСУВАННЯ:
  ├── Цифрові моделі рельєфу (ЦМР/DTM/DSM)
  ├── Топографічні карти 2.5D та 3D з ізолініями
  ├── Профілі траншей, доріг, трубопроводів по лінії
  ├── Розрахунок об'ємів земляних робіт (Cut & Fill)
  ├── Поперечні та поздовжні перерізи місцевості
  ├── Аналіз крутизни схилів (Slope / Aspect)
  ├── Водозбірні басейни та напрямки стоку
  └── Публікація картографічних видань

ДЖЕРЕЛА ДАНИХ:
  Copernicus DEM 25m (ESA) · SRTM 30m (NASA)
  USGS 3DEP 1m · LiDAR LAS/LAZ · Custom GeoTIFF
```

---

### ▸ 02 — ОБОРОНА ТА ТАКТИКА

```
ЗАСТОСУВАННЯ:
  ├── Тактичні симуляції на реальному рельєфі
  ├── Аналіз зон видимості (Viewshed / LOS)
  ├── Розрахунок мертвих зон і укриттів
  ├── Балістика: траєкторія снаряду з вітром і рельєфом
  ├── Артилерійські розрахунки по координатах
  ├── Планування маршрутів та обхід перешкод
  ├── Drone mission planning з перевіркою висоти
  ├── Зони ураження з урахуванням рельєфу
  ├── Тренажери та вправи (VBS-подібні системи)
  ├── Аналіз переправ і форсування водних перешкод
  ├── Offline режим: без інтернету в полі
  └── Класифіковані дані: E2E шифрування, air-gap

КЛЮЧОВІ МОДУЛІ:
  Viewshed Engine · Ballistics Solver · NavMesh A*
  Threat Zone Calculator · Route Optimizer
```

---

### ▸ 03 — МІСТОПЛАНУВАННЯ ТА УРБАНІСТИКА

```
ЗАСТОСУВАННЯ:
  ├── 3D генплан міста на реальному рельєфі
  ├── Digital Twin міста з live-даними
  ├── Аналіз щільності забудови та інсоляції
  ├── Тіньовий аналіз від майбутніх будівель
  ├── Шумова карта (noise pollution map)
  ├── Транспортні потоки та моделювання трафіку
  ├── Зелені зони, паркова інфраструктура
  ├── Зонування та регуляторні обмеження
  ├── Сценарії розвитку (що буде за 20 років?)
  └── Презентація проектів інвесторам та громаді

ІНТЕГРАЦІЇ:
  OpenStreetMap · CityGML LOD1-4 · Overture Maps
  GTFS транспорт · IoT сенсори · AQI дані
```

---

### ▸ 04 — БУДІВНИЦТВО ТА BIM

```
ЗАСТОСУВАННЯ:
  ├── Прив'язка BIM моделі до реального рельєфу
  ├── Планування будівельного майданчика
  ├── Розрахунок об'ємів земляних робіт
  ├── Аналіз доступу та логістики на майданчик
  ├── Тіньовий аналіз (інсоляція) нової будівлі
  ├── Видимість з вікон (права на вид)
  ├── Шумове забруднення від об'єкту
  ├── Моніторинг будівництва (план vs факт)
  └── AR preview будівлі на реальному місці

ФОРМАТИ:
  IFC · CityGML · glTF · Revit (через IFC)
  DXF / AutoCAD · Shapefile
```

---

### ▸ 05 — ЕКОЛОГІЯ ТА МОНІТОРИНГ ДОВКІЛЛЯ

```
ЗАСТОСУВАННЯ:
  ├── Зони затоплення при підйомі рівня ріки/моря
  ├── Моделювання поширення лісових пожеж
  ├── Моніторинг зсувів і нестабільних схилів
  ├── Ерозія берегів і русел річок
  ├── Зміни ландшафту у часі (порівняння знімків)
  ├── NDVI аналіз: стан рослинності (Sentinel-2)
  ├── Теплові острови в містах (LST)
  ├── Забруднення: поширення в повітрі та воді
  ├── Оцінка вуглецевого депо (ліс, ґрунт)
  └── Динаміка льодовиків і снігового покриву

ДАНІ:
  Sentinel-2 10m · Landsat 8/9 · Copernicus Services
  MODIS · Google Earth Engine · ESA Climate Data
```

---

### ▸ 06 — НАДЗВИЧАЙНІ СИТУАЦІЇ ТА РЯТУВАННЯ

```
ЗАСТОСУВАННЯ:
  ├── Планування маршрутів евакуації населення
  ├── Пошуково-рятувальні операції (SAR planning)
  ├── Зони ризику: повінь, зсув, хімічна аварія
  ├── Координація рятувальних служб на карті
  ├── Оптимальне розміщення укриттів
  ├── Розрахунок пропускної здатності доріг
  ├── Live треки рятувальних бригад (GPS)
  ├── Прогноз поширення хімічної хмари (CFD)
  └── Офлайн-робота без інтернету в зоні НС

REAL-TIME:
  WebSocket live feeds · GPS trackers · IoT sensors
  OpenWeatherMap · River gauges · Camera streams
```

---

### ▸ 07 — ІГРИ ТА ІНТЕРАКТИВНІ ЗАСТОСУНКИ

```
ЗАСТОСУВАННЯ:
  ├── Open World на реальних картах (будь-який регіон)
  ├── War simulator: реальні театри бойових дій
  ├── Flight simulator: реальний рельєф та міста
  ├── Racing: реальні треки та дороги
  ├── Survival / Exploration на реальній місцевості
  ├── Historical reconstruction
  ├── Battle Royale на реальних островах
  ├── RPG з реальною географією світу
  └── Metaverse із прив'язкою до координат

ІГРОВИЙ СТЕК:
  ECS · NavMesh A* · Behavior Trees · Physics (Rapier)
  GPU Instancing · LOD · Crowds Simulation
```

---

### ▸ 08 — КІНО, МЕДІА ТА АРХВІЗУАЛІЗАЦІЯ

```
ЗАСТОСУВАННЯ:
  ├── Превізуалізація локацій для зйомок
  ├── Планування drone-шотів з FPV симулятором
  ├── Архітектурна візуалізація в реальному оточенні
  ├── Cinematic keyframe анімація сцен
  ├── Virtual Production (LED wall контент)
  ├── 360° відео для VR перегляду
  ├── Наукова візуалізація (публікації, гранти)
  ├── Інтерактивні атласи та медіапродукти
  └── Time-lapse: місто, пори року, геологія

РЕНДЕР:
  PBR Materials · ACES Tonemapping · Volumetric Clouds
  Atmospheric Scattering · Cinematic DOF · Bloom
```

---

### ▸ 09 — НАУКА ТА ДОСЛІДЖЕННЯ

```
ГЕОЛОГІЯ:
  ├── 3D розрізи геологічних шарів
  ├── Тектонічні розломи та сейсмічність (USGS live)
  ├── Моделі родовищ та об'єм запасів
  └── Стабільність схилів (Mohr-Coulomb)

КЛІМАТОЛОГІЯ:
  ├── Кліматичні моделі ERA5, CMIP6
  ├── Зміна клімату: анімація 1950→2100
  ├── Підйом рівня моря: сценарії затоплення
  └── Льодовики: динаміка відступу

АСТРОНОМІЯ:
  ├── Реальне зоряне небо (astropy/Stellarium)
  ├── Траєкторії супутників: ISS, Starlink live
  ├── Сонячні та місячні затемнення
  └── Азимут сонця/місяця для будь-якої дати

ОКЕАНОГРАФІЯ:
  ├── GEBCO батиметрія — дно океану
  ├── Морські течії та хвилі (CMEMS/HYCOM)
  └── Температура поверхні океану (SST)
```

---

### ▸ 10 — ЕНЕРГЕТИКА ТА ІНФРАСТРУКТУРА

```
ЗАСТОСУВАННЯ:
  ├── Сонячний потенціал даху (PV planning)
  ├── Вітровий потенціал ділянки (Rose diagram)
  ├── Оптимальні маршрути ліній електропередач
  ├── 3D модель мереж водопостачання / каналізації
  ├── Прокладання трубопроводів (мінімальний рельєф)
  ├── Розміщення підстанцій та розподільних пунктів
  ├── Аварійне планування: де вразливі вузли
  └── Smart Grid: відображення live-навантажень

ІНТЕГРАЦІЇ:
  OpenStreetMap power= tags · IoT SCADA sensors
  Meteo API · Copernicus ERA5 wind data
```

---

### ▸ 11 — СІЛЬСЬКЕ ГОСПОДАРСТВО ТА АГРО

```
ЗАСТОСУВАННЯ:
  ├── NDVI карти врожайності (Sentinel-2, 10m)
  ├── Зони затоплення та дренажу полів
  ├── Аналіз схилів: де ерозія, де накопичення
  ├── Маршрути сільгосптехніки (оптимум)
  ├── Моніторинг посівів у часі (time-series)
  ├── Прогноз урожайності (ML на даних)
  ├── Карти внесення добрив (variable rate)
  └── Аналіз меліоративних систем

ДАНІ:
  Sentinel-2 NDVI · Landsat Thermal · Soil maps
  Weather stations · Drone multispectral
```

---

### ▸ 12 — ТРАНСПОРТ ТА ЛОГІСТИКА

```
ЗАСТОСУВАННЯ:
  ├── 3D моделювання доріг і розв'язок
  ├── Оптимізація маршрутів з урахуванням рельєфу
  ├── Моделювання трафіку в 3D
  ├── Планування залізничних трас (ухил, радіус)
  ├── Порти та морська логістика
  ├── Авіаційне планування (terrain clearance)
  ├── Last-mile delivery оптимізація
  └── Live: ADS-B літаки, AIS кораблі, GTFS транспорт

REAL-TIME FEEDS:
  ADS-B Exchange · MarineTraffic AIS
  OpenRailwayMap · GTFS-Realtime
```

---

### ▸ 13 — ДРОНИ ТА UAS

```
ЗАСТОСУВАННЯ:
  ├── 3D планування польотних місій
  ├── Перевірка terrain clearance по маршруту
  ├── Зони no-fly (аеропорти, об'єкти)
  ├── Розрахунок часу польоту та заряду АКБ
  ├── FPV симуляція до реального польоту
  ├── Автопілот маршрут по координатах
  ├── Ретрансляція live-відео з дрону на 3D карту
  ├── Бойові дрони: маршрут з обходом загроз
  └── Фотограмметрія: drone photos → 3D модель

СТАНДАРТИ:
  MAVLink · DroneKit · OpenSfM · COLMAP
```

---

### ▸ 14 — ВОДНІ РЕСУРСИ ТА ГІДРОЛОГІЯ

```
ЗАСТОСУВАННЯ:
  ├── Моделювання повені при різних рівнях води
  ├── Водозбірні басейни та водорозділи
  ├── Розрахунок дебіту річок (залежно від опадів)
  ├── Динаміка меандрування русел
  ├── Седиментація та накопичення наносів
  ├── Управління водосховищами (рівень, скидання)
  ├── Підземні води: рівень і напрямок потоку
  ├── Цунамі симуляція: поширення та затоплення
  └── Морська батиметрія (GEBCO) — дно океану

ДАНІ:
  HydroSHEDS · GEBCO · CMEMS · Argo Floats
  Gauge stations · OpenWeatherMap
```

---

### ▸ 15 — XR: VIRTUAL, AUGMENTED, MIXED REALITY

```
VIRTUAL REALITY (WebXR):
  ├── Immersive walk по реальному рельєфу
  ├── God mode: весь регіон у руці
  ├── VR Drawing та анотування у просторі
  ├── Multi-user VR: колеги в одній сцені
  └── Haptic feedback (вібрація = рельєф)

AUGMENTED REALITY:
  ├── Топографічна карта на столі (WebAR)
  ├── AR навігація по місту (стрілки + POI)
  ├── Майбутня будівля: AR preview на місці
  └── Геодезичні мітки в полі (AR точки)

MIXED REALITY:
  ├── Hololens / Magic Leap підтримка
  ├── Passthrough: реальний світ + 3D дані
  └── Collaborative MR для команд

ПЛАТФОРМИ:
  Meta Quest · Valve Index · Apple Vision Pro
  HoloLens 2 · Magic Leap 2 · Mobile AR
```

---

### ▸ 16 — ОСВІТА ТА НАУКОВА КОМУНІКАЦІЯ

```
ЗАСТОСУВАННЯ:
  ├── Інтерактивні 3D атласи та підручники
  ├── Навчальні симуляції (географія, геологія)
  ├── Візуалізація наукових даних (публікації)
  ├── Археологічна реконструкція місць
  ├── Астрономічні симуляції для планетаріїв
  ├── Кліматичні сценарії для wide-audience
  ├── Jupyter Notebook + вбудований 3D вьюпорт
  └── Моделі для музеїв та виставок

JUPYTER ІНТЕГРАЦІЯ:
  %load_ext geoengine.jupyter
  engine = GeoEngine(inline=True)
  engine.show()  # → 3D вьюпорт прямо в notebook
```

---

### ▸ 17 — ЦИФРОВИЙ ДВІЙНИК (DIGITAL TWIN)

```
LIVE ДАНІ:
  ├── IoT сенсори → точки на 3D карті (MQTT)
  ├── Метеостанції (температура, вологість, вітер)
  ├── Рівнеміри річок (попередження повені)
  ├── Якість повітря (PM2.5, CO2, NO2)
  ├── Сейсмографи (реальний час, USGS)
  └── Камери RTSP → проекція на 3D поверхню

МІСЬКИЙ ДВІЙНИК:
  ├── Комунальні мережі live (де аварія зараз)
  ├── Енергоспоживання будівель (теплова карта)
  ├── Трафік real-time + пробки
  ├── Будівельні проекти: план vs факт
  └── Демографія та щільність населення

ПРОМИСЛОВИЙ ДВІЙНИК:
  ├── Завод / кар'єр / порт у 3D
  ├── Техніка з GPS → живий рух на моделі
  └── Аварійне планування та evac routes
```

---

### ▸ 18 — ФОТОГРАММЕТРІЯ ТА 3D РЕКОНСТРУКЦІЯ

```
ПАЙПЛАЙН:
  Drone photos → Feature matching → SfM →
  Dense reconstruction → Mesh → Texture →
  GeoEngine scene (прив'язка до координат)

ЗАСТОСУВАННЯ:
  ├── 3D модель будівлі/об'єкту з дрону
  ├── Моніторинг деформацій (порівняння в часі)
  ├── Об'єм кар'єру / насипу (точний розрахунок)
  ├── Реставрація пам'яток архітектури
  ├── Документування руйнувань (збитки)
  ├── Accuracy report: похибка реконструкції
  └── Export: glTF · OBJ · LAS · GeoTIFF

ІНСТРУМЕНТИ:
  COLMAP · OpenSfM · ODM (OpenDroneMap)
  Gaussian Splatting · NeRF (instant-ngp)
```

---

### ▸ 19 — ПЛАНЕТИ ТА КОСМОС

```
ПІДТРИМУВАНІ ТІЛА:
  ├── Земля        — WGS84, всі джерела
  ├── Марс         — HiRISE DEM (25cm resolution!)
  ├── Місяць       — LOLA топографія NASA
  ├── Europa       — льодові кратери
  ├── Titan        — метанові озера Cassini
  └── Процедурні   — генерація за параметрами

КОСМІЧНІ ФУНКЦІЇ:
  ├── ISS та Starlink live (TLE orbit data)
  ├── Реальне зоряне небо (astropy)
  ├── Sunrise/sunset по координатах і даті
  ├── Сонячна система: орбіти та масштаб
  └── Видимість Чумацького Шляху (Bortle scale)
```

---

## ПОВНИЙ ПЕРЕЛІК МОДУЛІВ

```
CORE ENGINE              GEOSPATIAL               AI / ML
─────────────            ─────────────            ────────
Scene Graph              DEM Processing           Semantic Segmentation
Entity Component Sys     Tile Streaming           Object Detection
Asset Manager            Coordinate Transform     Change Detection
LOD System (CDLOD)       Heightmap → Mesh         Super Resolution
Frustum Culling          Vector Data (OSM)        AI Terrain Gen
Quadtree Adaptive        3D Buildings             Text → Scene
WebGPU Renderer          Road Networks            Photo → 3D (NeRF)
Physics (Rapier)         Water Bodies             SAM2 Integration
                         Land Cover Classification LLM NPC Agents

VISUALIZATION            SIMULATION               ANALYTICS
─────────────            ──────────               ─────────
PBR Materials            Flood Modeling           Viewshed Analysis
Atmospheric Scattering   Fire Spread              Slope / Aspect
Volumetric Clouds        Ballistics               Shadow Analysis
Dynamic Shadows (CSM)    Crowd Simulation         Volume Calc
Water / Ocean FFT        Vehicle Physics          Cross Sections
Vegetation System        Avalanche / Landslide    Noise Map
Post Processing          Seismic Waves            Heat Map
Time of Day              Wind CFD (lite)          Flow Maps
Weather System           Erosion Simulation       Buffer Zones

REAL-TIME                XR                       PLATFORM
─────────                ──────                   ────────
WebSocket Streams        WebXR VR/AR              Python API
IoT / MQTT               VR Multi-user            JavaScript API
GPS Trackers             AR Marker-based          Plugin System
ADS-B (Aircraft)         Mixed Reality            Jupyter Widget
AIS (Ships)              Haptic Feedback          CLI Tools
GTFS-RT (Transit)        Spatial Audio            REST API
Camera RTSP              Passthrough MR           gRPC Streaming
Live Weather             Apple Vision Pro         Offline / PWA
```

---

## ПОРІВНЯННЯ З КОНКУРЕНТАМИ

| Можливість | GeoEngine | Three.js | Cesium.js | Panda3D | Unreal |
|---|:---:|:---:|:---:|:---:|:---:|
| Python-first API | ✅ | ❌ | ❌ | ⚠️ | ❌ |
| Реальні геодані з коробки | ✅ | ❌ | ✅ | ❌ | ❌ |
| WebGPU рендер | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| AI / ML інтеграція | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| Фізичні симуляції | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| VR / AR підтримка | ✅ | ⚠️ | ⚠️ | ❌ | ✅ |
| Digital Twin / IoT | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| Відкритий код | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| Offline / Air-gap | ✅ | ⚠️ | ❌ | ✅ | ✅ |
| Планетарний масштаб | ✅ | ❌ | ✅ | ❌ | ⚠️ |
| Jupyter інтеграція | ✅ | ❌ | ❌ | ❌ | ❌ |
| Plugin ecosystem | ✅ | ✅ | ⚠️ | ⚠️ | ✅ |

---

## АРХІТЕКТУРА API

### Python

```python
from geoengine import Scene, Terrain, Buildings, Camera, Simulation

# Завантажити реальний рельєф Карпат
scene = Scene()
terrain = Terrain.from_bbox(
    bbox=(48.0, 23.0, 48.5, 24.0),
    source="copernicus",      # ESA Copernicus DEM
    resolution=10             # метрів на піксель
)
scene.add(terrain)

# Будівлі з OpenStreetMap
buildings = Buildings.from_osm(terrain.bbox, lod=2)
scene.add(buildings)

# Аналіз видимості
viewshed = terrain.compute_viewshed(
    observer=(48.25, 23.5, 10.0),   # lat, lon, висота
    max_distance=8000                 # метри
)
scene.add_layer(viewshed, color="heatmap")

# Симуляція повені
flood = Simulation.flood(terrain, water_level=650)  # метрів
scene.add_layer(flood, opacity=0.7)

scene.camera.fly_to(lat=48.25, lon=23.5, altitude=3000)
scene.render()
```

### JavaScript

```javascript
import { GeoEngine, Terrain, Camera, Viewshed } from '@geoengine/core'

const engine = new GeoEngine({ canvas: '#viewport', renderer: 'webgpu' })

const terrain = await Terrain.fromBBox({
  bbox: [48.0, 23.0, 48.5, 24.0],
  source: 'copernicus',
  lod: 'auto'                   // автоматичний LOD
})

engine.scene.add(terrain)

engine.camera.flyTo({
  lat: 48.25, lon: 23.5,
  altitude: 2000,
  duration: 3000                 // анімація 3 секунди
})

// Real-time IoT дані
engine.iot.connect('wss://sensors.example.com')
engine.iot.on('reading', (sensor) => {
  engine.scene.updateMarker(sensor.id, sensor.value)
})
```

---

## ТЕХНІЧНІ ХАРАКТЕРИСТИКИ

```
ПРОДУКТИВНІСТЬ:
  Terrain:    до 10,000 км² без просідань FPS
  Buildings:  1,000,000+ об'єктів (GPU instancing)
  Particles:  5,000,000+ частинок (Compute Shader)
  Physics:    10,000 rigid bodies @ 60fps (Rapier)
  AI Agents:  100,000 crowd agents (GPU)

ТОЧНІСТЬ:
  Координати: до 1 мм у локальній СК
  Висоти:     1-25m залежно від джерела DEM
  Фото-3D:    до 1 см (LiDAR або HDR фото)

МАСШТАБ:
  Мінімум:    1 мм (інтер'єр, деталь)
  Максимум:   весь глобус (планетарний режим)
  Перехід:    плавний LOD без стрибків

ФОРМАТИ ВХІД:
  Растр:  GeoTIFF · HGT · ASC · IMG · NC
  Вектор: GeoJSON · Shapefile · KML · GML · DXF
  3D:     glTF · OBJ · FBX · IFC · CityGML · LAS
  Фото:   JPG · PNG · TIFF · RAW · GeoTIFF (SAT)

ФОРМАТИ ВИХІД:
  3D:     glTF 2.0 · OBJ · FBX · USDZ · STL
  Гео:    GeoTIFF · GeoJSON · Shapefile · KML · DXF
  Медіа:  PNG · EXR · MP4 · WebM · 360° відео
  Звіти:  PDF · HTML · JSON analytics
```

---

## ЛІЦЕНЗІЯ ТА УМОВИ

```
© [Chugaister8] [2026]. Всі права захищені.

Цей код є приватною власністю. Будь-яке використання, копіювання, модифікація або поширення без явної письмової згоди автора **заборонено**.

© [Chugaister8] [2026]. All rights reserved.

This code is proprietary. No permission is granted to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the software without explicit prior written permission from the author.
```

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GEOENGINE  ·  The Geospatial OS  ·  v0.1-alpha
  Python · JavaScript · WebGPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

         "The Earth deserves better software."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
