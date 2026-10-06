# Публикация документации

Сайт собирается из Markdown-файлов в `docs/` с помощью MkDocs Material.
Навигация, тема и проверка ссылок заданы в `mkdocs.yml`; версии инструментов — в `docs/requirements.txt`.
Датированные отчёты проверки и исходные результаты хранятся в репозитории.

## Локальный просмотр

Из корня репозитория:

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs serve
```

Откройте `http://127.0.0.1:8000`. Изменения Markdown появятся после сохранения файла.
Инструменты документации устанавливаются в отдельное окружение uv.

Проверить сборку перед публикацией:

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

Готовый сайт появится в `site/`. Неисправные ссылки, якоря и страницы навигации приводят к ошибке сборки.

## GitHub Pages

1. Разместите проект в [репозитории ZarrSwarm](https://github.com/dsuhoi/zarrswarm). Для GitHub Free Pages доступны в публичных
   репозиториях; поддержка приватных зависит от тарифа.
2. В **Settings → Pages → Build and deployment → Source** выберите **GitHub Actions**.
3. Добавьте файлы документации, `mkdocs.yml` и `.github/workflows/docs.yml` в ветку по умолчанию.
   Первый запуск можно сделать вручную: **Actions → Documentation → Run workflow**.

Workflow проверяет сборку при изменениях документации в push и pull request. Публикация выполняется
из ветки по умолчанию; адрес сайта появляется в окружении `github-pages` и в результате шага deployment.
Адрес документации ZarrSwarm: `https://dsuhoi.github.io/zarrswarm/`.

В CI адрес берётся из GitHub Pages автоматически. Для локальной проверки конкретного адреса:

```bash
DOCS_SITE_URL=https://dsuhoi.github.io/zarrswarm/ \
  uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

В разрешённых ветках окружения `github-pages` должна быть ветка по умолчанию.
Дополнительный токен для этого workflow не требуется: используются `GITHUB_TOKEN` и OIDC GitHub Actions.

Официальная инструкция: [GitHub Pages с собственным workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
