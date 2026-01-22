from typing import Any, Dict, Type
from pydantic import BaseModel
import json


class PydanticSchemaUtils:
    """
    Static utilities for converting Pydantic models into
    LLM-friendly JSON example schemas.

    - Respects Field(exclude=True)
    - Respects model_config['exclude']
    - Uses field descriptions
    - Produces stable, prompt-safe output
    """

    @staticmethod
    def to_descriptive_json(
        model: Type[BaseModel],
        *,
        include_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a JSON example skeleton from a Pydantic model.
        """
        schema = model.model_json_schema()
        excluded_fields = PydanticSchemaUtils._collect_excluded_fields(model)

        return PydanticSchemaUtils._build_example_from_schema(
            schema=schema,
            excluded_fields=excluded_fields,
            include_descriptions=include_descriptions,
        )

    @staticmethod
    def to_descriptive_pretty_json(
        model: Type[BaseModel],
        *,
        include_descriptions: bool = True,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> str:
        """
        Generate a pretty-printed JSON example string from a Pydantic model.
        """
        example = PydanticSchemaUtils.to_descriptive_json(
            model,
            include_descriptions=include_descriptions,
        )

        return json.dumps(
            example,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )

    # ==========================================================
    # Internal helpers
    # ==========================================================

    @staticmethod
    def _build_example_from_schema(
        *,
        schema: Dict[str, Any],
        excluded_fields: set[str],
        include_descriptions: bool,
    ) -> Any:
        schema_type = schema.get("type")

        # ---------- Objects ----------
        if schema_type == "object":
            result = {}
            properties = schema.get("properties", {})

            for field_name, field_schema in properties.items():
                # 🔒 Skip excluded fields
                if field_name in excluded_fields:
                    continue

                example = PydanticSchemaUtils._build_example_from_schema(
                    schema=field_schema,
                    excluded_fields=set(),
                    include_descriptions=include_descriptions,
                )

                if include_descriptions and "description" in field_schema:
                    result[field_name] = {
                        "_description": field_schema["description"],
                        "_value": example,
                    }
                else:
                    result[field_name] = example

            return result

        # ---------- Arrays ----------
        if schema_type == "array":
            items_schema = schema.get("items", {})
            return [
                PydanticSchemaUtils._build_example_from_schema(
                    schema=items_schema,
                    excluded_fields=set(),
                    include_descriptions=include_descriptions,
                )
            ]

        # ---------- Primitives ----------
        return PydanticSchemaUtils._primitive_placeholder(schema)

    @staticmethod
    def _primitive_placeholder(schema: Dict[str, Any]) -> Any:
        t = schema.get("type")

        if t == "string":
            return "string"
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return False

        return None

    @staticmethod
    def _collect_excluded_fields(model: Type[BaseModel]) -> set[str]:
        excluded: set[str] = set()

        # Field-level exclude=True
        for name, field in model.model_fields.items():
            if field.exclude:
                excluded.add(name)

        # Model-level exclude config
        model_exclude = model.model_config.get("exclude")
        if isinstance(model_exclude, (set, list, tuple)):
            excluded.update(model_exclude)

        return excluded