"""FastAPI application providing a UI for the job role analyzer."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from job_role_analyzer.data_models import JobRoleWithCompetencies

from .dependencies import get_analyzer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Job Role Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class AnalyzeRequest(BaseModel):
    job_title: str
    job_description: str
    years_of_experience: int


class CompetencyDTO(BaseModel):
    name: str
    level: int
    type: str | None


class AnalyzeResponse(BaseModel):
    job_role_id: str
    normalized_job_role_summary: str
    competencies: list[CompetencyDTO]


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, analyzer=Depends(get_analyzer)) -> AnalyzeResponse:
    try:
        result: JobRoleWithCompetencies = analyzer.analyze(
            job_title=request.job_title,
            job_description=request.job_description,
            years_of_experience=request.years_of_experience,
        )
    except ValueError as exc:  # pragma: no cover - runtime validation
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_role = result.job_role
    return AnalyzeResponse(
        job_role_id=str(job_role.job_role_id),
        normalized_job_role_summary=job_role.normalized_summary,
        competencies=[
            CompetencyDTO(name=item.name, level=item.level, type=item.type)
            for item in result.competencies
        ],
    )


@app.on_event("shutdown")
async def close_dependencies() -> None:
    analyzer = get_analyzer()
    analyzer.db.close()
    llm_client = analyzer.llm_interface.client
    close = getattr(llm_client, "close", None)
    if callable(close):
        close()
