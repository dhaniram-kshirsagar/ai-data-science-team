Step 1: 

┌─────────────────────────────────────────────────────────────────────┐
│ KGInsights                                    👤 User ▼            │
├─────────────────────────────────────────────────────────────────────┤
│ [Dashboard] | Generate Schema | Manage Schema | Display KG | Insights│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐  ┌───────────────────────┐                 │
│  │ + New Knowledge Graph│  │ 🔍 Search Graphs...   │                 │
│  └─────────────────────┘  └───────────────────────┘                 │
│                                                                     │
│  Knowledge Graphs                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Name        | Description      | Created      | Actions         ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Customer    | Customer-product | 2025-03-05   | Manage Schema   ││
│  │ Relations   | relationships    |              | Insights        ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Supply      | Supply chain     | 2025-02-20   | Manage Schema   ││
│  │ Chain       | network          |              | Insights        ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  Available Datasets for KG Generation                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Dataset    | Preview | Schema | Generate KG                     ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Sales Q1   |   👁️    |   📋   |    ➕                          ││
│  │ HR Data    |   👁️    |   📋   |    ➕                          ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Step 2: 

On clicking New KG or + button, get associated file path and call build schema.

Step 3: 

Build Schema's output should be displayed as below:


┌─────────────────────────────────────────────────────────────────────┐

│ Generate Schema                                👤 User ▼            │
├─────────────────────────────────────────────────────────────────────┤
│ Dashboard | [Generate Schema] | Manage Schema | Insights            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Select Dataset: [Sales Q1_______▼]  ┌────────────────────┐         │
│                                      │ Save Schema Version│         │
│                                      └────────────────────┘         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Recommended Schema:                                             │ │
│  │                                                                 │ │
│  │ ┌─────────────────────────────────────────────────────────────┐ │ │
│  │ │                                                             │ │ │
│  │ │ [Visualization of suggested graph schema with nodes         │ │ │
│  │ │  and edges]                                                 │ │ │
│  │ │                                                             │ │ │
│  │ │                                                             │ │ │
│  │ └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                 │ │
│  │ Nodes:                                                          │ │
│  │ - Customer (id, name, segment)                                  │ │
│  │ - Product (id, name, category)                                  │ │
│  │ - Sale (id, date, amount)                                       │ │
│  │                                                                 │ │
│  │ Relationships:                                                  │ │
│  │ - PURCHASED (Customer → Product)                                │ │
│  │ - CONTAINS (Sale → Product)                                     │ │
│  │ - MADE_BY (Sale → Customer)                                     │ │
│  │                                                                 │ │
│  │ Indexes:                                                        │ │
│  │ - Customer(id)                                                  │ │
│  │ - Product(id)                                                   │ │
│  │ - Sale(id)                                                      │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Step 4:

Save the schema

On clicking Save schema, save the schema in JSON file and show entry in below screen.

┌─────────────────────────────────────────────────────────────────────┐
│ KGInsights                                    👤 User ▼            │
├─────────────────────────────────────────────────────────────────────┤
│ [Dashboard] | Generate Schema | Manage Schema | Display KG | Insights│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐  ┌───────────────────────┐                 │
│  │ + New Knowledge Graph│  │ 🔍 Search Graphs...   │                 │
│  └─────────────────────┘  └───────────────────────┘                 │
│                                                                     │
│  Knowledge Graphs                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Name        | Description      | Created      | Actions         ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Customer    | Customer-product | 2025-03-05   | Manage Schema   ││
│  │ Relations   | relationships    |              | Insights        ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Supply      | Supply chain     | 2025-02-20   | Manage Schema   ││
│  │ Chain       | network          |              | Insights        ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  Available Datasets for KG Generation                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Dataset    | Preview | Schema | Generate KG                     ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ Sales Q1   |   👁️    |   📋   |    ➕                          ││
│  │ HR Data    |   👁️    |   📋   |    ➕                          ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
