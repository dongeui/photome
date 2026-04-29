"""Person management API — list, rename, and query face clusters."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from app.api.deps import require_state
from app.models.face import Face
from app.models.person import Person

router = APIRouter(prefix="/people", tags=["people"])


class PersonResponse(BaseModel):
    id: int
    display_name: str
    face_count: int
    sample_file_ids: list[str]


class RenamePersonRequest(BaseModel):
    display_name: str


@router.get("", response_model=list[PersonResponse])
def list_people(request: Request) -> list[PersonResponse]:
    """List all known persons with face counts."""
    database = require_state(request, "database")
    with database.session_factory() as session:
        rows = session.execute(
            select(Person, func.count(Face.id).label("face_count"))
            .outerjoin(Face, Face.person_id == Person.id)
            .group_by(Person.id)
            .order_by(func.count(Face.id).desc())
        ).all()
        result: list[PersonResponse] = []
        for person, face_count in rows:
            sample_faces = session.scalars(
                select(Face).where(Face.person_id == person.id).limit(3)
            ).all()
            result.append(
                PersonResponse(
                    id=person.id,
                    display_name=person.display_name,
                    face_count=face_count,
                    sample_file_ids=[str(f.file_id) for f in sample_faces],
                )
            )
        return result


@router.get("/{person_id}", response_model=PersonResponse)
def get_person(person_id: int, request: Request) -> PersonResponse:
    database = require_state(request, "database")
    with database.session_factory() as session:
        person = session.get(Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")
        face_count = session.scalar(
            select(func.count()).select_from(Face).where(Face.person_id == person_id)
        ) or 0
        sample_faces = session.scalars(
            select(Face).where(Face.person_id == person_id).limit(3)
        ).all()
        return PersonResponse(
            id=person.id,
            display_name=person.display_name,
            face_count=face_count,
            sample_file_ids=[str(f.file_id) for f in sample_faces],
        )


@router.patch("/{person_id}", response_model=PersonResponse)
def rename_person(
    person_id: int,
    body: RenamePersonRequest,
    request: Request,
) -> PersonResponse:
    """Update the display name for a person (face cluster)."""
    database = require_state(request, "database")
    new_name = body.display_name.strip()
    if not new_name:
        raise HTTPException(status_code=422, detail="display_name must not be empty")
    with database.session_factory() as session:
        person = session.get(Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")
        person.display_name = new_name
        session.commit()
        session.refresh(person)
        face_count = session.scalar(
            select(func.count()).select_from(Face).where(Face.person_id == person_id)
        ) or 0
        sample_faces = session.scalars(
            select(Face).where(Face.person_id == person_id).limit(3)
        ).all()
        return PersonResponse(
            id=person.id,
            display_name=person.display_name,
            face_count=face_count,
            sample_file_ids=[str(f.file_id) for f in sample_faces],
        )


@router.get("/{person_id}/media", response_model=list[str])
def list_person_media(
    person_id: int,
    request: Request,
    limit: int = 50,
) -> list[str]:
    """Return file_ids of media containing this person."""
    database = require_state(request, "database")
    with database.session_factory() as session:
        person = session.get(Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")
        file_ids = session.scalars(
            select(Face.file_id)
            .where(Face.person_id == person_id)
            .distinct()
            .limit(limit)
        ).all()
        return [str(fid) for fid in file_ids]
