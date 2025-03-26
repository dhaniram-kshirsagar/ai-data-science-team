# Graph Schema Generator API Documentation

## Overview
This API provides endpoints for generating Neo4j graph schemas from CSV data. It uses LLM-based analysis to infer schema structure and relationships.

## Endpoints

### 1. POST /upload
- **Purpose**: Upload a CSV file for schema generation
- **Request Body`:
  ```json
  {
    "file": "CSV file data"
  }
  ```
- **Response`:
  ```json
  {
    "message": "File uploaded successfully",
    "path": "/path/to/temp/file.csv"
  }
  ```
- **Errors`:
  - 500: Internal server error during file processing

### 2. POST /generate-schema
- **Purpose`: Generate Neo4j schema from uploaded CSV
- **Response`:
  ```json
  {
    "message": "Schema generated successfully"
  }
  ```
- **Errors`:
  - 400: No file uploaded
  - 500: Internal server error during schema generation

### 3. GET /get-schema
- **Purpose`: Retrieve generated schema and Cypher statements
- **Response`:
  ```json
  {
    "schema": {
      "nodes": [],
      "relationships": []
    },
    "cypher": "CREATE ...",
    "graph_image": "base64 encoded image"
  }
  ```
- **Errors`:
  - 400: No file uploaded
  - 404: Schema not generated
