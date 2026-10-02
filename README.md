# zarr-torrent

Децентрализованная P2P-раздача Zarr-массивов (ERA5 и любых CF-датасетов) между кластерами.
Трекера нет: Kademlia DHT, узлы за NAT ходят через релей. Реплики с разными периодами, наборами переменных,
чанками, кодеками и форматами (v2/v3) складываются в один рой и открываются как один `xarray.Dataset`.

## Установка

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e .
```

## Сеть

```bash
# первый публичный узел (bootstrap + релей), нужен открытый TCP-порт
zt init --bootstrap-node --public-host data.example.org     # печатает ztnet://<id>@data.example.org:7881
zt node

# остальные узлы (кластер, ноутбук, контейнер без входящих соединений)
zt init --join ztnet://<id>@data.example.org:7881 [--service]
zt node                                                    # или: systemctl --user enable --now zt-node
```

Настройки узла — в `~/.zt/config.toml` (создаёт `zt init`, с комментариями): что раздавать при старте
(`[[seed]] path = "/data/era5/*.zarr"`), потолок отдачи, доверенные ключи, тонкая настройка.

Гайды: [администратору](docs/operator.md) · [сеть, доступ, регистрация датасетов, ограничения](docs/network.md) ·
[пользователю](docs/user.md) · [справочник по конфигу](docs/config.md).

Закрытая сеть: `zt init --bootstrap-node --public-host HOST --private` печатает приглашение `ztnet://…?k=<ключ>`;
без ключа узлы сети не отвечают никому.

## Раздача и скачивание

```bash
zt seed /data/era5.zarr                      # -> zt://<grid>[+<grid>...]  (по подсетке на сигнатуру измерений)
zt search 2m_temperature                     # индекс метаданных в DHT: где лежит, у скольких сидов, охват по времени
zt get zt://… --vars t2m --time 2020-01-01:2020-03-01 --out t2m.zarr
zt mean zt://… t2m --rel-err 0.001           # среднее с доверительным интервалом, скачав только часть данных
zt name era5 zt://…                          # -> zt://era5@<pubkey>, подписанное изменяемое имя
zt status | zt peers zt://…
zt-tui                                       # клиент как торрент: загрузки, раздачи, метаданные, карта частей, настройки
```

## xarray

```python
import zarr_torrent as zt
ds = zt.open_dataset("zt://…")                                         # объединение всех реплик
ts = zt.open_dataset("zt://…", chunking={"time": -1, "lat": 1, "lon": 1})  # любая раскладка, собирается на лету
zt.prefetch(ds.t2m.sel(time=slice("2020-01", "2020-02")))              # один оптимальный план на весь срез
zt.progressive_mean_vas(ds, "t2m", rel_err=0.001)
```

## Доверие

Каждый чанк проверяется по хэшу байтов и по хэшу значений (`vcid`), значение выбирается большинством держателей.
Чтобы исключить сговор, укажите доверенных издателей: `zt node --trust <pubkey>`. Управляющий API
(`127.0.0.1:7882`) принимает только локальных клиентов с заголовком `X-Zt-Client: 1`.

## Разработка

```bash
.venv/bin/pytest tests                       # e2e-рой, матрица сценариев, безопасность, TUI, алгоритмы
.venv/bin/python sim/simulate.py --peers 24  # симулятор: стратегии, отказы, DHT, JLPS, потери
```

Архитектура, алгоритмы (JLPS, max-flow по сигнатурам доступности, VAS) и результаты — в `DESIGN.md`.
