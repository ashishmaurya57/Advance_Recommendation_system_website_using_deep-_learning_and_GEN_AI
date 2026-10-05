# BookTown

A book store with a hybrid recommendation engine (LLM relevance scoring, sentence
embeddings, sentiment analysis and interaction history).

| Part | Stack |
| --- | --- |
| `frontend/` | React 19, TypeScript, Vite, TanStack Router + Query, Tailwind CSS v4 |
| `backend/` | FastAPI, SQLAlchemy 2, Alembic, SQLAdmin, transformers / sentence-transformers, Groq |
| Database | Supabase Postgres |
| Files | Supabase Storage bucket `booktown` (covers, category images, profile photos, PDFs), indexed in the `media_files` table |

## Run it locally

Prerequisites: [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
# 1. Backend (http://localhost:8000)
cd backend
cp .env.example .env          # then fill in the values
uv sync
uv run alembic upgrade head   # create/upgrade tables in Supabase
uv run uvicorn app.main:app --reload --port 8000

# 2. Frontend (http://localhost:5173), in a second terminal
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` and `/admin` to the backend.

- **Admin panel:** http://localhost:5173/admin, sign in with the superuser from the old Django admin.
- **API docs:** http://localhost:8000/docs
- The first request that needs the ML models downloads them (about 350 MB, once).

## Project layout

```
backend/
  app/
    main.py            app setup: sessions, CORS, admin, routers
    core/              settings (.env), password hashing, auth dependencies
    db/                engine/session and ORM models
    schemas/           request/response models
    api/routes/        auth, profile, catalog, shop (cart/orders/payments), discover (search, recommendations), reader, contact
    services/          ml, recommender, search, interactions, pdf, storage (Supabase), files (upload + media_files)
    admin/             SQLAdmin views
  migrations/          Alembic migrations
  scripts/             one-off scripts (SQLite -> Supabase migration)
  tests/               pytest suite (runs on a throwaway SQLite DB, storage faked)
frontend/
  src/
    routes/            pages (file-based routing)
    components/        UI building blocks
    lib/               API client, queries, auth guard, i18n, payments
```

## Common tasks

```bash
cd backend
uv run pytest                                          # backend tests
uv run alembic revision --autogenerate -m "describe"   # after changing models
uv run alembic upgrade head

cd frontend
npm run build                                          # type-check + production build
```

## Data migration from the old Django app

The original SQLite data and uploaded files were copied to Supabase with
`uv run python -m scripts.migrate_from_sqlite` (sources in `backend/legacy_data/`, git-ignored).
Old customer passwords were stored in plain text; they are re-hashed automatically the next
time each customer signs in.
