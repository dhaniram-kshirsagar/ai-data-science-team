# Graph Schema API Design

## Step 1: Build Schema

### Requirements:
- Takes path of uploaded file
- Expects uploaded file and GraphSchemaAgent created with it
- Returns Schema and Cypher

### API Endpoint:
`POST /build-schema`

### Request:
```json
{
  "file_path": "/path/to/uploaded/file.csv"
}
```

### Response:
```json
{
  "schema": {
    "nodes": [],
    "relationships": [],
    "indexes": []
  },
  "cypher": "CREATE ..."
}
```

## Step 2: Enhance GraphSchemaAgent

### Requirements:
- Modify GraphSchemaAgent to handle multiple CSV files
- Update API to accept directory paths

### API Endpoint:
`POST /build-schema-from-directory`

### Request:
```json
{
  "directory_path": "/path/to/uploaded/files"
}
```

### Response:
```json
{
  "schema": {
    "nodes": [],
    "relationships": [],
    "indexes": []
  },
  "cypher": "CREATE ..."
}