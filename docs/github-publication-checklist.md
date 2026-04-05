# GitHub Publication Checklist

Use this checklist to publish `popcomplex` as a public repository safely.

## 1) Preflight

- Confirm generated outputs are excluded (`runs/`, `dist/`, `node_modules/`, `.env*`).
- Confirm no credentials are present in tracked files.
- Verify the app still builds:

```bash
npm run build
```

## 2) Initialize Git locally (first time only)

```bash
git init
git add .
git commit -m "Initial public release"
```

## 3) Create GitHub repository

Using GitHub CLI (`gh`) from this project root:

```bash
gh auth login
gh repo create popcomplex --public --source=. --remote=origin --push
```

If the repo already exists, connect and push:

```bash
git remote add origin git@github.com:<your-account>/popcomplex.git
git branch -M main
git push -u origin main
```

## 4) Repository settings (recommended)

- Add a repository description and topic tags.
- Enable branch protection for `main`.
- Enable Dependabot security updates.
- Add a `LICENSE` file (MIT/Apache-2.0/etc.) before wide sharing.

## 5) Post-publish checks

- Confirm `README.md` renders correctly.
- Confirm clone/install/build works in a clean environment.
- Open one issue for known limitations and one for roadmap.
