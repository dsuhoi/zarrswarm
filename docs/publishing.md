# Publish the documentation

MkDocs Material builds the website from Markdown in `docs/`.
Navigation, theme and link validation are configured in `mkdocs.yml`;
tool versions are pinned in `docs/requirements.txt`.

## Preview locally

From the repository root:

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs serve
```

Open `http://127.0.0.1:8000`. The preview reloads when Markdown changes.
Documentation tools run in a separate uv environment.

Check the build before publishing:

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

The generated site is written to `site/`. Missing navigation pages, local links
and anchors fail the strict build.

## GitHub Pages

1. Push the project to the [repository](https://github.com/dsuhoi/zarrswarm).
2. Open **Settings → Pages → Build and deployment → Source** and select
   **GitHub Actions**.
3. Push documentation, `mkdocs.yml` and `.github/workflows/docs.yml` to the
   default branch. To run it manually, select
   **Actions → Documentation → Run workflow**.

The workflow checks documentation changes on pushes and pull requests.
Deployment uses the default branch. The published address appears in the
`github-pages` environment and the deployment step.
The configured site address is `https://dsuhoi.github.io/zarrswarm/`.

In CI, the address is read from GitHub Pages. For a local check at a particular
address, set `DOCS_SITE_URL` before the build:

```bash
export DOCS_SITE_URL=https://dsuhoi.github.io/zarrswarm/
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

The `github-pages` environment must allow deployment from the default branch.
The workflow uses GitHub Actions' `GITHUB_TOKEN` and OIDC; no additional
deployment token is needed. Enabling Pages in repository settings requires
repository administration access.

See the [official GitHub Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
