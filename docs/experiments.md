# Гайд: воспроизвести эксперименты статьи

Каждый эксперимент — скрипт в `bench/` или `sim/`, который пишет JSON с результатами; числа в статье берутся из
этих файлов (иногда через сводный скрипт). Скрипты запускаются из корня репозитория интерпретатором `.venv`.
Значение по умолчанию у `--out` — путь, под которым результат лежит в репозитории; чтобы не затереть его, задайте
свой.

## Актуальные результаты рукописи

| Место | Данные / результат |
|---|---|
| Таблица 1 | `bench/lattice_probe.json`: binary survey; `bench/revalidation/provider_identity_controls_v5.json`: рабочий L3 на тех же четырёх variables и shared-quantizer controls |
| Таблица 2 | `bench/revalidation/provider_merge_v5.json`, `decoder_survey_v5.json`, `goes_decoders_v5.json`; контроли в `lattice_controls_v5.json` |
| Таблица 3 | `bench/revalidation/multisite_v5_final.json`: 54 WAN measurements предыдущего L3 snapshot |
| Таблица 4 и график ablations | `process_fibonacci_v5.json`, `paired_stats_v5.json`, `kernel_v5.json`, `kernel_stats_v5.json`; source hashes записаны в provenance, kernel comparison содержит два полностью записанных размещения |
| Таблица 5 | `edge_v5.json`, `metadata_v5.json`; дополнительные identifier lengths в `identifier_size_v5.json` |
| Joint-cover controls (§3.4) | `planner_cover_oracle_v5.json`: 240 fixed graphs, 120 development и 120 проверочных; полный перебор covers/holders и parameter sensitivity |
| Assignment correctness | `assignment_controls_v5.json`: существующий exhaustive test, 600 fixed-cover cases |

Ниже сохранены исторические команды и прежние имена файлов. Они не обозначают нынешние номера таблиц и не подменяют результаты L3.

## 1. Историческая карта экспериментов

| эксп. | что | скрипт | результат |
|---|---|---|---|
| Табл. 1 | 9 публичных копий ERA5 | `bench/replica_survey.py [--time 2010-03-03T06]` | `bench/replica_survey.json` |
| Табл. 2 | ARCO vs NCAR, 4 переменные × 3 момента | `bench/lattice_probe.py` | `bench/lattice_probe.json` |
| E0 | слияние двух провайдеров в рое | `bench/real_provider_merge.py WORKDIR` | `bench/real_provider_merge.json` |
| E0, контроли | оценщик системы по тайлам + квантованные негативные контроли | `bench/lattice_controls.py WORKDIR` (WORKDIR с `arco.zarr`/`ncar.zarr` из E0) | `bench/lattice_controls.json` |
| E1/E2 | 36 узлов по трассам Meteor-M, 3 запроса; lattice+JLPS / lattice+min-bytes / byte identity, 10 повторов | `sim/e_hetero.py`, `bench/summarize_ehet.py` | `sim/results_ehet_v5.json` (части `_a`…`_d`), парная статистика `sim/results_ehet_v5_stats.json` |
| свип E3 | время против числа узлов 12/24/36/48 | `sim/e_hetero.py --peers N`, `bench/summarize_sweep.py` | `sim/results_sweep5_{12,24,36,48}.json` |
| E3 | DHT на 24/64/100 узлах | `sim/simulate.py --only e4 [--peers N]` | `sim/results_e234.json`, `sim/results_scale64.json`, `sim/results_scale100.json` |
| E4 | доступность по окнам видимости (Монте-Карло) | `bench/sat_traces.py`, `bench/avail_mc.py` | `bench/avail_mc_meteor.json`, `paper/figs/plots/avail_vs_p.csv` |
| E5 | CPU-стоимость проверки на 1 vCPU | `bench/bench_edge_cost.py sample/measure` | `bench/edge_cost_panel_lattice.json` |
| E6 | GOES-16 ABI L1b, 3 декодера | `bench/goes_decoders.py DIR` | `bench/goes_decoders.json` |
| E6 | fMRI OpenNeuro ds000102 | `bench/fmri_demo.py WORKDIR` | `bench/fmri_demo.json` |
| обзор декодеров | MRMS, GFS, GOES-16, ERA5 | `bench/decoder_survey.py GRIB_DIR` | `bench/decoder_survey.json` |
| E7 | 3 площадки + HTTP-зеркала по каталогу + абляция зондирования | `sim/multisite.py sim/multisite.toml`, `sim/http_mirrors.py`, `bench/summarize_multisite.py` | `sim/results_multisite_final.json` |
| E8 | E1/E2 на эмуляции уровня ядра | `sim/e_hetero.py --emu` и `--procs`, `bench/compare_emulators.py` | `sim/e8/e8_{emu,procs}_{0,1,2}.json`, `bench/emulators.json` |
| память | RSS узла, размер метаданных | `bench/bench_memory.py REPLICA.zarr`, `bench/bench_metadata.py` | `bench/memory*.json`, `bench/metadata.json` |

Быстрые проверки без роя:

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m zarr_torrent.plan && .venv/bin/python -m zarr_torrent.jlps && .venv/bin/python -m zarr_torrent.parity
.venv/bin/python bench/lattice_controls.py ~/.cache/zt_providers   # нужны arco.zarr/ncar.zarr из real_provider_merge
.venv/bin/python bench/goes_decoders.py ~/.cache/zt_goes           # 3 файла GOES-16 ABI-L1b-RadC C13
```

## 2. Данные для E1/E2/E7/E8

```bash
# пять вариантов хранения ОДНИХ значений (ERA5 2 m temperature из ARCO): V1-arco-1h, V2-wb2-6h, V3-ts-1h,
# V4-tile-6h, V5-tiles-1h — разные шаг, чанки, кодеки, zarr v2/v3
.venv/bin/python bench/make_variants.py ~/zt_ms/variants_month --start 2020-01-01 --days 31
# окна видимости наземных станций SatNOGS для спутников Meteor-M (skyfield, текущие TLE)
.venv/bin/python bench/sat_traces.py --start 2020-01-01 --days 31 --stations 40 --sats METEOR-M2 \
    --out bench/sat_traces_meteor.json
.venv/bin/python bench/avail_mc.py --traces bench/sat_traces_meteor.json --out bench/avail_mc_meteor.json   # E4
```

## 3. Три стенда

Все три запускают настоящие процессы `zt node` (`python -m zarr_torrent.cli node`), у каждого свои сокеты.
`ProcSwarm` и `EmuSwarm` имеют один интерфейс (`nodes`, `meta`, `ctl`, `kill`, `stop`), поэтому `sim/e_hetero.py`
работает на любом из них без изменений. Рой: `--boot` bootstrap+релеев (полоса 25 МБ/с), остальные — держатели;
доля `--nat` держателей за NAT ходит через случайный релей; полоса отдачи держателя логнормальная вокруг медианы,
обрезанная до ×0.2…×5; односторонняя задержка 2–40 мс. Номер повтора задаёт размещение (зерно), так что повтор r
одинаков на всех стендах.

### Уровень процессов: `sim/procswarm.py` (`ProcSwarm`)

Все узлы на `127.0.0.1`. Полоса отдачи — token bucket внутри узла (`--upload-mbps`), задержку добавляет сам узел
(`ZT_EMU_LATENCY_MS`). TCP, очереди и потери ядра при этом не моделируются. По умолчанию для E1–E4 в статье.

### Уровень ядра: `sim/netemu.py` (`EmuSwarm`)

Как Mininet, но без root: драйвер перезапускает себя внутри `unshare -rn` (пользовательское + сетевое пространство
имён, внутри он root) и становится «маршрутизатором». В нём L2-мост `br0` (10.0.255.254/16, сеть 10.0.0.0/16); каждый
узел запускается в своём `unshare -n`, соединяется с мостом парой veth, получает адрес 10.0.x.y/16. На **обоих**
концах veth стоят `tc netem` (задержка, потери) и под ним `tbf` (скорость): от узла — его полоса отдачи, к узлу —
max(полоса отдачи, 50 МБ/с); у узла без заданной полосы (клиент) — 30 МБ/с в обе стороны. Так TCP, очереди шейпера и повторные передачи — ядра.
Узел объявляет полосу в DHT (`ZT_ANNOUNCE_MBPS`), но сам её не ограничивает и задержку не добавляет; управляющий
API слушает адрес узла (`ZT_CTL_HOST`). Маршрутизация и запись в `/proc/sys` не нужны (подходит для контейнеров).

Один раз на хосте (root):

```bash
modprobe -a veth sch_netem sch_tbf          # нужны и непривилегированные user namespaces
.venv/bin/python sim/netemu.py selftest     # 2 узла, 1 МБ/с и 50 мс: значения совпали, время соответствует шейпингу
```

### Реальные площадки: `sim/multisite.py` + `sim/multisite.toml`

Локально — bootstrap+релей и свежий клиент на каждый запрос; держатели — на удалённых площадках (`[sites.*]`:
`host` для ssh, `python`, `pythonpath`/`src` с кодом, `work`, `nice`), список `[[holders]]` (первый — «одно
зеркало»), запросы `[queries.*]`, эталон `truth`. К каждой площадке поднимается обратный ssh-туннель
`ssh -R BP:127.0.0.1:BP`, где BP — порт локального bootstrap: удалённый порт **равен** локальному, поэтому адрес
релея `http://127.0.0.1:BP/r/<id>`, который объявляет удалённый узел, верен на обоих концах. Удалённые узлы
запускаются по ssh (живут, пока жива сессия), прогревают кэш страниц своей реплики и раздают её. Сеть закрытая:
ключ генерируется на каждый запуск (`ZT_NETWORK_KEY`).

Фазы (`--phases`, по умолчанию `cloud,mirror,swarm,bytes`):
`cloud` — xarray читает регион прямо из бакета ARCO (GCS); `http` — базовая линия `sim/http_mirrors.py`: клиент по
каталогу (`[[http_mirrors]]`: площадка, путь, `first_hour`, `hours`) делит файлы чанков между зеркалами
`python -m http.server` на площадках (через `ssh -L`, 8 запросов в полёте на зеркало) и ничего не проверяет;
`mirror` — один держатель; `swarm` — все держатели, покрытия `jlps` и `bytes` (min-bytes); `bytes` — byte identity,
клиент пробует каждый байтовый рой и берёт лучший (оракул).

```bash
.venv/bin/python sim/multisite.py sim/multisite.toml --phases cloud,http,mirror,swarm,bytes --reps 3 \
    --out sim/results_multisite_new.json
.venv/bin/python bench/summarize_multisite.py sim/results_multisite_new.json   # таблица + строки LaTeX
```

В `sim/results_multisite_final.json` фазы `cloud`/`mirror`/`http` — из первого прогона, `swarm`/`bytes` — с
зондированием первого контакта; строки без него — в ключе `ablation_without_probing`.

## 4. `sim/e_hetero.py`: E1/E2, свип, E8

```
python sim/e_hetero.py VARIANTS_DIR [флаги]
```

| флаг | по умолчанию | смысл |
|---|---|---|
| `--procs` | выкл. | стенд уровня процессов (`ProcSwarm`) |
| `--emu` | выкл. | стенд уровня ядра (`EmuSwarm`); без обоих — узлы в одном процессе (`simulate.Swarm`) |
| `--placement` | `random` | `random` — окна вариантов; `sat` — первые `--stations` узлов — наземные станции с часами из `--traces`, остальные — зеркала |
| `--peers` | 48 | **всего** узлов, включая `--boot` bootstrap-релеев (держателей = peers − boot) |
| `--boot`, `--nat` | 3, 0.3 | число bootstrap+релеев, доля держателей за NAT |
| `--rep-start`, `--reps` | 0, 3 | повторы `range(rep_start, reps)`: **`--reps` — конец диапазона, а не число** |
| `--modes` | `values,bytes` | идентичность: по значениям (наша) и byte identity (`ZT_IDENTITY=bytes`) |
| `--covers` | `jlps` | покрытия в режиме values; `jlps,bytes` добавляет min-bytes (E2) |
| `--queries` | `map_day_1h,series_point_1h,period_6h` | карта за сутки, точечный ряд за весь период, весь период с шагом 6 ч |
| `--rate-mbps` | 4.0 | медиана полосы отдачи держателя |
| `--traces`, `--stations` | `bench/sat_traces.json`, 24 | трассы и число станций для `--placement sat` |
| `--out` | `sim/results_ehet.json` | файл результатов (пишется после каждой строки: прерванный прогон сохраняет сделанное) |

Рабочий каталог роёв — `ZT_SIM_ROOT` (по умолчанию `~/.cache/zt_sim`); `ZT_KEEP_LOGS=1` сохраняет логи узлов.

```bash
V=~/zt_ms/variants_month
COMMON="--procs --placement sat --traces bench/sat_traces_meteor.json --peers 36 --stations 22 --rate-mbps 1"
# E1/E2: 10 повторов (в статье повтор 0 последовательно, 1–9 тремя параллельными окнами)
.venv/bin/python sim/e_hetero.py $V $COMMON --covers jlps,bytes --rep-start 0 --reps 1 --out sim/results_ehet_new_a.json
.venv/bin/python sim/e_hetero.py $V $COMMON --covers jlps,bytes --rep-start 1 --reps 4 --out sim/results_ehet_new_b.json
# ... --rep-start 4 --reps 7, --rep-start 7 --reps 10; затем строки `rows` объединяются в один файл
.venv/bin/python bench/summarize_ehet.py sim/results_ehet_new.json --csv e1e2_new.csv
# свип: N = 12, 24, 36, 48 (станций 7, 14, 22, 29)
.venv/bin/python sim/e_hetero.py $V --procs --placement sat --traces bench/sat_traces_meteor.json --rate-mbps 1 \
    --peers 12 --stations 7 --queries map_day_1h,period_6h --out sim/results_sweep_12.json
.venv/bin/python bench/summarize_sweep.py sim/results_sweep_{12,24,36,48}.json
# E8: те же первые три роя на двух стендах одного хоста
.venv/bin/python sim/e_hetero.py $V $COMMON --covers jlps,bytes --reps 3 --out sim/e8_procs.json
.venv/bin/python sim/e_hetero.py $V ${COMMON/--procs/--emu} --covers jlps,bytes --reps 3 --out sim/e8_emu.json
.venv/bin/python bench/compare_emulators.py sim/e8_procs.json sim/e8_emu.json --reps 3 --out bench/emulators.json
```

Объединить части в один файл (сводные скрипты принимают один файл на конфигурацию):

```bash
.venv/bin/python -c "import json,sys; f=sys.argv[2:]; d=json.load(open(f[0])); \
d['rows']=[r for p in f for r in json.load(open(p))['rows']]; json.dump(d, open(sys.argv[1],'w'), indent=1)" OUT.json PART...
```

Сводные скрипты:
- `bench/summarize_ehet.py RESULTS [--csv paper/figs/plots/e1e2.csv]` — по запросу и конфигурации медиана/мин/макс
  секунд, медиана МБ, полнота;
- `bench/summarize_sweep.py FILE...` — медиана времени по идентичности и запросу против числа узлов →
  `paper/figs/plots/speed_vs_peers.csv`;
- `bench/summarize_multisite.py RESULTS` — медиана [мин, макс] секунд, МБ, раскладки, худшая ошибка против эталона;
- `bench/compare_emulators.py PROC EMU [--out bench/emulators.json] [--reps N]` — медианы по конфигурациям на обоих
  стендах, парные ускорения lattice+JLPS и совпадает ли ранжирование конфигураций.

Парную статистику строит `bench/summarize_ehet.py RESULTS --stats OUTPUT.json`: медиана отношений времени
в полных парах, 95% bootstrap-интервал (10 000 повторов, seed 0), число полных пар и более быстрых запусков.
Старый `sim/results_ehet_v5_stats.json` относится к прежней версии оценщика; `v5` в этом имени — версия
эксперимента, а не актуальный формат `lattice-v5`. Текущие повторные результаты хранятся отдельно в
`bench/revalidation/`; подробности исправлений — в `docs/CORRECTNESS_REVALIDATION.md`.


В повторной проверке `lattice-v5` сетевые строки допускаются в итог только при `state=done`, полном
покрытии запрошенных samples и `value_check=exact`. Прямая WAN-проверка использует `max_abs_err=0`.
Источник сравнения — уже скачанные payload через `/api/read`; reference comparison выполняется после
измерения времени. Кэш завершённого клиента удаляется сразу после проверки, чтобы серия не накапливала
копии данных до заполнения NFS quota. Remote driver записывает PID узла и сверяет точный `--home`
перед TERM, а при необходимости — перед KILL после пяти секунд ожидания.


## Исправления повторной проверки L3

Рукопись использует файлы `bench/revalidation/`, а не прежние E-серии выше. Текущий драйвер
не ограничивает поиск полного byte answer временем частичного ответа. После первого полного
ответа последующие кандидаты получают `max(120 s, 3 × best complete time)`; до него — 1800 s.
Coverage хранится без округления: частичный длинный ряд не должен округляться до единицы.
Если выбранный набор чанков не покрывает запрос, Node один раз обновляет каталог и повторяет
планирование. Время этого повтора входит в query timing; настоящие пробелы остаются `partial`.
Source snapshots до и после этих изменений сохранены раздельно. Их численные результаты
нельзя объединять с утверждением, что всё измерено на одной окончательной копии исходников.

JLPS исключает cached chunks из сетевого lower bound и нормировки remote-price loads; минимальный по
байтам cover всегда входит в кандидаты. Гарантия относится к оценкам одной модели с заданным slack,
а не к измеренному wall time. Таблицы process/kernel повторяются с исправленным solver; WAN сохраняет
свою измеренную версию. Файлы прежних снимков не выдаются за замеры текущего solver.
