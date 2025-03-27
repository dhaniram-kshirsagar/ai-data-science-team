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

### 2. GET /build-schema
- **Purpose**: Generate Neo4j schema from uploaded CSV and return results
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
  - 400: No file uploaded
  - 404: Schema generation failed
  - 500: Internal server error during schema generation

/Graph Model
Upload CSV 
Build Schema
  getFiles already uploaded

Schema Management
Save API
Apply Database