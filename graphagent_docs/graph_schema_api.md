# Graph Schema Generator API Documentation

## Overview
This API provides endpoints for generating Neo4j graph schemas from CSV data. It uses LLM-based analysis to infer schema structure and relationships.

## Endpoints

### 1. POST /build-schema
- **Purpose**: Generate Neo4j schema from a CSV file at a specified absolute path
- **Request Body`:
  ```json
  {
    "file_path": "/absolute/path/to/file.csv"
  }
  ```
- **Response`:
  ```json
  {
    "schema": {
      "nodes": [
        {
          "label": "NodeType",
          "properties": [
            {
              "name": "propertyName",
              "type": "dataType",
              "constraints": ["constraintType"]
            }
          ]
        }
      ],
      "relationships": [],
      "indexes": [
        {
          "label": "NodeType",
          "properties": ["propertyName"]
        }
      ]
    },
    "cypher": "CREATE (...) // Generated Cypher statements",
    "graph_image": "base64_encoded_image"
  }
  ```
- **Errors`:
  - 400: Only CSV files are supported
  - 404: File not found at specified path
  - 500: Schema generation failed

### 2. POST /save-schema
- **Purpose**: Save the generated schema JSON to a file on the server
- **Request Body`:
  ```json
  {
    "schema": {
      // Schema object
      // Same object received from /build-schema endpoint
    },
    "output_path": "/optional/path/to/save/schema.json" // Optional
  }
  ```
- **Response`:
  ```json
  {
    "message": "Schema saved successfully",
    "file_path": "/path/where/schema/was/saved.json"
  }
  ```
- **Errors`:
  - 500: Failed to save schema

## Notes
- If no output path is provided to the /save-schema endpoint, the schema will be saved in the server's default location (./saved_schemas) with a timestamp-based filename.
- The API automatically creates any necessary directories when saving a schema to a specified path.
- All schema files are saved in JSON format with proper indentation for readability.