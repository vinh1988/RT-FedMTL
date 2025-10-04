# 🏗️ Streaming Federated Learning Architecture

## Overview: Two Approaches Comparison

This document provides a comprehensive architectural view of our streaming federated learning implementations, comparing the **LoRA-enabled** and **No-LoRA** approaches.

---

## 🎯 High-Level Architecture Comparison

```mermaid
graph TB
    subgraph "🚀 With LoRA (Parameter Efficient)"
        subgraph "🌐 Federated Server"
            S1[Global Model<br/>BERT-base + LoRA<br/>109M params]
            S2[Aggregation Engine<br/>FedAvg Algorithm]
            S3[WebSocket Server<br/>Communication Hub<br/>Port 8766]
            S4[Parameter Serializer<br/>JSON Encoder/Decoder]
            S5[Knowledge Distillation<br/>Teacher Inference]
        end
        
        subgraph "👤 Client 1: SST-2"
            C1A[Local Model<br/>Tiny-BERT + LoRA<br/>4.4M + 16K params]
            C1B[WebSocket Client<br/>Connection Handler]
            C1C[Local Training Loop<br/>SST-2 Dataset + KD Loss]
            C1D[Parameter Manager<br/>Deserialize → Update Model]
            C1E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        subgraph "👤 Client 2: QQP"
            C2A[Local Model<br/>Tiny-BERT + LoRA<br/>4.4M + 16K params]
            C2B[WebSocket Client<br/>Connection Handler]
            C2C[Local Training Loop<br/>QQP Dataset + KD Loss]
            C2D[Parameter Manager<br/>Deserialize → Update Model]
            C2E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        subgraph "👤 Client 3: STS-B"
            C3A[Local Model<br/>Tiny-BERT + LoRA<br/>4.4M + 16K params]
            C3B[WebSocket Client<br/>Connection Handler]
            C3C[Local Training Loop<br/>STS-B Dataset + KD Loss]
            C3D[Parameter Manager<br/>Deserialize → Update Model]
            C3E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        %% Server Internal Flow
        S1 --> S4
        S1 --> S5
        S4 --> S3
        S5 --> S3
        S3 --> S2
        
        %% Server to Client Flow
        S3 -->|📤 LoRA Params<br/>(33K serialized)| C1B
        S3 -->|📤 Teacher Logits<br/>(KD data)| C1B
        S3 -->|📤 LoRA Params<br/>(33K serialized)| C2B
        S3 -->|📤 Teacher Logits<br/>(KD data)| C2B
        S3 -->|📤 LoRA Params<br/>(33K serialized)| C3B
        S3 -->|📤 Teacher Logits<br/>(KD data)| C3B
        
        %% Client Internal Flow After WebSocket Receive
        C1B -->|1. Deserialize JSON| C1D
        C1D -->|2. Update LoRA weights| C1A
        C1A -->|3. Updated model| C1C
        C1C -->|4. Training metrics| C1E
        C1E -->|5. Serialize results| C1B
        
        C2B -->|1. Deserialize JSON| C2D
        C2D -->|2. Update LoRA weights| C2A
        C2A -->|3. Updated model| C2C
        C2C -->|4. Training metrics| C2E
        C2E -->|5. Serialize results| C2B
        
        C3B -->|1. Deserialize JSON| C3D
        C3D -->|2. Update LoRA weights| C3A
        C3A -->|3. Updated model| C3C
        C3C -->|4. Training metrics| C3E
        C3E -->|5. Serialize results| C3B
        
        %% Client to Server Flow
        C1B -->|📥 Updated LoRA<br/>(16K serialized)| S3
        C2B -->|📥 Updated LoRA<br/>(16K serialized)| S3
        C3B -->|📥 Updated LoRA<br/>(16K serialized)| S3
        
        %% Server Aggregation Flow
        S3 -->|6. Collect updates| S2
        S2 -->|7. Aggregate LoRA| S1
    end
    
    subgraph "⚡ Without LoRA (Full Training)"
        subgraph "🌐 Federated Server"
            NS1[Global Model<br/>BERT-base Full<br/>109M params]
            NS2[Aggregation Engine<br/>FedAvg Algorithm]
            NS3[WebSocket Server<br/>Communication Hub<br/>Port 8768]
            NS4[Parameter Serializer<br/>JSON Encoder/Decoder]
            NS5[Knowledge Distillation<br/>Teacher Inference]
        end
        
        subgraph "👤 Client 1: SST-2"
            NC1A[Local Model<br/>Tiny-BERT Full<br/>4.4M params]
            NC1B[WebSocket Client<br/>Connection Handler]
            NC1C[Local Training Loop<br/>SST-2 Dataset + KD Loss]
            NC1D[Parameter Manager<br/>Deserialize → Update Model]
            NC1E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        subgraph "👤 Client 2: QQP"
            NC2A[Local Model<br/>Tiny-BERT Full<br/>4.4M params]
            NC2B[WebSocket Client<br/>Connection Handler]
            NC2C[Local Training Loop<br/>QQP Dataset + KD Loss]
            NC2D[Parameter Manager<br/>Deserialize → Update Model]
            NC2E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        subgraph "👤 Client 3: STS-B"
            NC3A[Local Model<br/>Tiny-BERT Full<br/>4.4M params]
            NC3B[WebSocket Client<br/>Connection Handler]
            NC3C[Local Training Loop<br/>STS-B Dataset + KD Loss]
            NC3D[Parameter Manager<br/>Deserialize → Update Model]
            NC3E[Metrics Collector<br/>Loss, Accuracy, etc.]
        end
        
        %% Server Internal Flow
        NS1 --> NS4
        NS1 --> NS5
        NS4 --> NS3
        NS5 --> NS3
        NS3 --> NS2
        
        %% Server to Client Flow
        NS3 -->|📤 Compatible Params<br/>(~1M serialized)| NC1B
        NS3 -->|📤 Teacher Logits<br/>(KD data)| NC1B
        NS3 -->|📤 Compatible Params<br/>(~1M serialized)| NC2B
        NS3 -->|📤 Teacher Logits<br/>(KD data)| NC2B
        NS3 -->|📤 Compatible Params<br/>(~1M serialized)| NC3B
        NS3 -->|📤 Teacher Logits<br/>(KD data)| NC3B
        
        %% Client Internal Flow After WebSocket Receive
        NC1B -->|1. Deserialize JSON| NC1D
        NC1D -->|2. Update all weights| NC1A
        NC1A -->|3. Updated model| NC1C
        NC1C -->|4. Training metrics| NC1E
        NC1E -->|5. Serialize results| NC1B
        
        NC2B -->|1. Deserialize JSON| NC2D
        NC2D -->|2. Update all weights| NC2A
        NC2A -->|3. Updated model| NC2C
        NC2C -->|4. Training metrics| NC2E
        NC2E -->|5. Serialize results| NC2B
        
        NC3B -->|1. Deserialize JSON| NC3D
        NC3D -->|2. Update all weights| NC3A
        NC3A -->|3. Updated model| NC3C
        NC3C -->|4. Training metrics| NC3E
        NC3E -->|5. Serialize results| NC3B
        
        %% Client to Server Flow
        NC1B -->|📥 Full Model<br/>(4.4M serialized)| NS3
        NC2B -->|📥 Full Model<br/>(4.4M serialized)| NS3
        NC3B -->|📥 Full Model<br/>(4.4M serialized)| NS3
        
        %% Server Aggregation Flow
        NS3 -->|6. Collect updates| NS2
        NS2 -->|7. Aggregate compatible| NS1
    end
```

### 🔍 **Detailed Flow: What Happens After WebSocket Receives Data**

```
🎯 COMPLETE DATA FLOW AFTER WEBSOCKET RECEIVE:

📥 Client Receives from Server:
┌─────────────────────────────────────────────────────────────┐
│ 1️⃣ WebSocket Client receives JSON message                    │
│    └── Contains: {'type': 'train_start', 'parameters': {...}}│
│                                                             │
│ 2️⃣ Parameter Manager deserializes JSON → PyTorch tensors    │
│    └── JSON arrays → torch.Tensor objects                   │
│                                                             │
│ 3️⃣ Local Model gets parameter update                        │
│    ├── LoRA: Update adapter weights (16K params)            │
│    └── No-LoRA: Update all compatible weights (4.4M params) │
│                                                             │
│ 4️⃣ Training Loop uses updated model                         │
│    ├── Forward pass with new parameters                     │
│    ├── Compute task loss + knowledge distillation loss     │
│    └── Backward pass & gradient updates                     │
│                                                             │
│ 5️⃣ Metrics Collector gathers training results              │
│    └── Loss, accuracy, convergence metrics                  │
│                                                             │
│ 6️⃣ Results serialized back to JSON                         │
│    └── Updated parameters + metrics → JSON format          │
│                                                             │
│ 7️⃣ WebSocket Client sends back to server                   │
│    └── {'type': 'train_complete', 'parameters': {...}}      │
└─────────────────────────────────────────────────────────────┘

📤 Server Receives from Clients:
┌─────────────────────────────────────────────────────────────┐
│ 6️⃣ WebSocket Server collects all client updates            │
│    └── Waits for all 3 clients to complete training        │
│                                                             │
│ 7️⃣ Aggregation Engine performs FedAvg                      │
│    ├── LoRA: Average LoRA adapter weights                   │
│    └── No-LoRA: Average compatible parameters only          │
│                                                             │
│ 8️⃣ Global Model updated with aggregated parameters         │
│    └── Ready for next federated learning round             │
└─────────────────────────────────────────────────────────────┘
```

### 📊 **Component Interaction Details**

| **Component** | **Role After WebSocket Receive** | **LoRA Version** | **No-LoRA Version** |
|---------------|-----------------------------------|------------------|---------------------|
| **Parameter Manager** | Deserialize & apply updates | Update 16K LoRA weights | Update 4.4M full weights |
| **Local Model** | Use updated parameters | BERT + LoRA adapters | Full Tiny-BERT |
| **Training Loop** | Train with new knowledge | Task + KD loss | Task + KD loss |
| **Metrics Collector** | Gather performance data | LoRA training metrics | Full training metrics |
| **WebSocket Client** | Send results back | 16K updated params | 4.4M updated params |

---

## 📡 WebSocket Communication Architecture

### 🔧 **Correct Understanding: WebSocket as Communication Layer**

**WebSocket is NOT between models directly** - it's the **communication infrastructure** between server and client **processes**:

```mermaid
graph LR
    subgraph "🌐 Server Process"
        SP1[Federated Server<br/>Python Process]
        SP2[Global Model<br/>BERT-base]
        SP3[WebSocket Server<br/>asyncio + websockets]
        SP4[Aggregation Logic<br/>FedAvg Algorithm]
    end
    
    subgraph "📡 Communication Layer"
        WS1[WebSocket Protocol<br/>TCP/IP Network]
        WS2[JSON Serialization<br/>Parameter Transfer]
        WS3[Real-time Streaming<br/>Bidirectional]
    end
    
    subgraph "👤 Client Process 1"
        CP1[Federated Client<br/>Python Process]
        CP2[Local Model<br/>Tiny-BERT]
        CP3[WebSocket Client<br/>Connection Handler]
        CP4[Training Loop<br/>Local Updates]
    end
    
    SP1 --> SP2
    SP1 --> SP3
    SP1 --> SP4
    SP3 <-->|Network<br/>Communication| WS1
    WS1 <--> WS2
    WS2 <--> WS3
    WS3 <-->|Network<br/>Communication| CP3
    CP3 --> CP1
    CP1 --> CP2
    CP1 --> CP4
    
    style WS1 fill:#2196f3
    style WS2 fill:#2196f3
    style WS3 fill:#2196f3
```

### 🎯 **What Actually Communicates via WebSocket**

```mermaid
graph TD
    subgraph "📤 Server → Client Messages"
        S2C1[Welcome Message<br/>Connection confirmation]
        S2C2[Training Start<br/>Round initialization]
        S2C3[Global Parameters<br/>Serialized weights]
        S2C4[Teacher Logits<br/>Knowledge distillation]
        S2C5[Round Complete<br/>Acknowledgment]
    end
    
    subgraph "📥 Client → Server Messages"
        C2S1[Registration<br/>Client ID + task info]
        C2S2[Status Updates<br/>Training progress]
        C2S3[Local Parameters<br/>Updated weights]
        C2S4[Training Metrics<br/>Loss & accuracy]
        C2S5[Completion Signal<br/>Round finished]
    end
    
    subgraph "🔄 WebSocket Channel"
        WS[Bidirectional<br/>Real-time Stream<br/>JSON Messages]
    end
    
    S2C1 --> WS
    S2C2 --> WS
    S2C3 --> WS
    S2C4 --> WS
    S2C5 --> WS
    
    WS --> C2S1
    WS --> C2S2
    WS --> C2S3
    WS --> C2S4
    WS --> C2S5
    
    style WS fill:#4caf50
```

### 🏗️ **Layered Architecture: Separation of Concerns**

```mermaid
graph TB
    subgraph "🎯 Application Layer"
        AL1[Federated Learning Logic<br/>Training rounds, aggregation]
        AL2[Model Management<br/>Parameter updates, KD]
    end
    
    subgraph "📊 Business Layer"
        BL1[Training Coordination<br/>Client synchronization]
        BL2[Parameter Serialization<br/>Weight encoding/decoding]
    end
    
    subgraph "📡 Communication Layer"
        CL1[WebSocket Server/Client<br/>Connection management]
        CL2[Message Protocol<br/>JSON formatting]
    end
    
    subgraph "🌐 Network Layer"
        NL1[TCP/IP Protocol<br/>Reliable transmission]
        NL2[Operating System<br/>Socket management]
    end
    
    AL1 --> BL1
    AL2 --> BL2
    BL1 --> CL1
    BL2 --> CL2
    CL1 --> NL1
    CL2 --> NL2
    
    style CL1 fill:#ff9800
    style CL2 fill:#ff9800
```

### ✅ **Key Clarification: WebSocket Role**

```
🔍 CORRECT Understanding:

WebSocket Communication Flow:
┌─────────────────┐    WebSocket     ┌─────────────────┐
│  Server Process │ ◄─────────────► │  Client Process │
│                 │   (Network)      │                 │
│ ┌─────────────┐ │                  │ ┌─────────────┐ │
│ │Global Model │ │                  │ │Local Model │ │
│ │BERT-base    │ │                  │ │Tiny-BERT   │ │
│ └─────────────┘ │                  │ └─────────────┘ │
└─────────────────┘                  └─────────────────┘

❌ INCORRECT: "WebSocket between models"
✅ CORRECT: "WebSocket between server and client processes"

The models themselves don't communicate - the federated learning 
processes use WebSocket to exchange:
├── 📤 Serialized parameters (model weights as JSON)
├── 📥 Training instructions and coordination
├── 📊 Metrics and status updates
└── 🎓 Knowledge distillation data (teacher logits)
```

### 🎯 **Real Implementation Example**

```python
# Server side (fixed_streaming_glue.py):
async def client_handler(self, websocket):
    # WebSocket handles the CONNECTION, not the model
    registration = await websocket.recv()  # Receive from client process
    
    # The MODELS are separate objects:
    server_model = self.model  # BERT-base model
    parameters = server_model.get_parameters()  # Extract weights
    
    # WebSocket sends SERIALIZED parameters:
    await websocket.send(json.dumps({
        'type': 'train_start',
        'parameters': self.serialize_parameters(parameters)  # Model → JSON
    }))

# Client side (fixed_streaming_glue.py):
async def run_client(self):
    async with websockets.connect(uri) as websocket:
        # WebSocket handles CONNECTION to server process
        
        message = await websocket.recv()  # Receive from server process
        data = json.loads(message)
        
        if data['type'] == 'train_start':
            # DESERIALIZE parameters back to model:
            server_params = self.deserialize_parameters(data['parameters'])  # JSON → Model
            self.model.set_parameters(server_params)  # Update local model
```

---

## 🔄 Detailed Training Flow Architecture

### 🎨 With LoRA: Parameter-Efficient Approach

```mermaid
sequenceDiagram
    participant Server as 🌐 Server<br/>(BERT-base + LoRA)
    participant WS as 📡 WebSocket Hub
    participant C1 as 👤 Client SST-2<br/>(Tiny-BERT + LoRA)
    participant C2 as 👤 Client QQP<br/>(Tiny-BERT + LoRA)
    participant C3 as 👤 Client STS-B<br/>(Tiny-BERT + LoRA)
    
    Note over Server,C3: 🚀 Round 1: LoRA Training
    
    Server->>WS: Initialize global LoRA parameters (33K params)
    WS->>C1: Send LoRA weights + teacher logits
    WS->>C2: Send LoRA weights + teacher logits
    WS->>C3: Send LoRA weights + teacher logits
    
    Note over C1,C3: Local Training (LoRA only)
    C1->>C1: Train LoRA adapters<br/>SST-2 data + KD loss
    C2->>C2: Train LoRA adapters<br/>QQP data + KD loss
    C3->>C3: Train LoRA adapters<br/>STS-B data + KD loss
    
    C1->>WS: Updated LoRA params (16K)
    C2->>WS: Updated LoRA params (16K)
    C3->>WS: Updated LoRA params (16K)
    
    WS->>Server: Aggregate LoRA parameters
    Server->>Server: FedAvg on LoRA weights only
    
    Note over Server,C3: 🔄 Next Round...
```

### ⚡ Without LoRA: Full Parameter Training

```mermaid
sequenceDiagram
    participant Server as 🌐 Server<br/>(BERT-base Full)
    participant WS as 📡 WebSocket Hub
    participant C1 as 👤 Client SST-2<br/>(Tiny-BERT Full)
    participant C2 as 👤 Client QQP<br/>(Tiny-BERT Full)
    participant C3 as 👤 Client STS-B<br/>(Tiny-BERT Full)
    
    Note over Server,C3: 🚀 Round 1: Full Training
    
    Server->>WS: Initialize full parameters (109M params)
    WS->>C1: Send compatible params + teacher logits
    WS->>C2: Send compatible params + teacher logits
    WS->>C3: Send compatible params + teacher logits
    
    Note over C1,C3: Local Training (Full Model)
    C1->>C1: Train all parameters<br/>SST-2 data + KD loss
    C2->>C2: Train all parameters<br/>QQP data + KD loss
    C3->>C3: Train all parameters<br/>STS-B data + KD loss
    
    C1->>WS: Updated full params (4.4M)
    C2->>WS: Updated full params (4.4M)
    C3->>WS: Updated full params (4.4M)
    
    WS->>Server: Aggregate compatible parameters
    Server->>Server: FedAvg on overlapping weights
    
    Note over Server,C3: 🔄 Next Round...
```

---

## 🧠 Knowledge Distillation Architecture

### 📚 Cross-Architecture Learning Flow

```mermaid
graph LR
    subgraph "🎓 Teacher Model (Server)"
        T1[BERT-base<br/>12 layers<br/>768 hidden<br/>109M params]
        T2[Teacher Logits<br/>Task-specific outputs]
    end
    
    subgraph "🎯 Knowledge Transfer"
        KD1[Temperature Scaling<br/>T = 4.0]
        KD2[Distillation Loss<br/>α = 0.7]
        KD3[Task Loss<br/>α = 0.3]
    end
    
    subgraph "👨‍🎓 Student Models (Clients)"
        S1[Tiny-BERT<br/>2 layers<br/>128 hidden<br/>4.4M params]
        S2[Student Logits<br/>Task-specific outputs]
    end
    
    T1 --> T2
    T2 --> KD1
    KD1 --> KD2
    S1 --> S2
    S2 --> KD3
    KD2 --> S1
    KD3 --> S1
    
    style T1 fill:#e1f5fe
    style S1 fill:#f3e5f5
    style KD2 fill:#fff3e0
```

---

## 💾 Data Flow & Parameter Management

### 🔄 LoRA Parameter Flow

```mermaid
graph TD
    subgraph "LoRA Parameter Lifecycle"
        A[Base Model<br/>Frozen Parameters<br/>4.4M params] --> B[LoRA Adapters<br/>Trainable Parameters<br/>16K params]
        B --> C[Forward Pass<br/>Base + LoRA outputs]
        C --> D[Loss Computation<br/>Task + KD losses]
        D --> E[Backward Pass<br/>LoRA gradients only]
        E --> F[Parameter Update<br/>LoRA weights only]
        F --> G[Serialize LoRA<br/>16K params → JSON]
        G --> H[WebSocket Transfer<br/>Lightweight: ~64KB]
        H --> I[Server Aggregation<br/>FedAvg on LoRA only]
        I --> J[Broadcast Updated<br/>LoRA parameters]
        J --> B
    end
    
    style B fill:#4caf50
    style F fill:#4caf50
    style H fill:#2196f3
```

### ⚡ Full Parameter Flow

```mermaid
graph TD
    subgraph "Full Parameter Lifecycle"
        A[Full Model<br/>All Parameters<br/>4.4M params] --> B[Forward Pass<br/>Complete model]
        B --> C[Loss Computation<br/>Task + KD losses]
        C --> D[Backward Pass<br/>All gradients]
        D --> E[Parameter Update<br/>All 4.4M weights]
        E --> F[Serialize Full Model<br/>4.4M params → JSON]
        F --> G[WebSocket Transfer<br/>Heavy: ~18MB]
        G --> H[Server Aggregation<br/>Compatible params only]
        H --> I[Broadcast Compatible<br/>Parameters]
        I --> A
    end
    
    style E fill:#ff9800
    style G fill:#f44336
    style H fill:#9c27b0
```

---

## 🏛️ System Architecture Components

### 🔧 Core Components Breakdown

```mermaid
graph TB
    subgraph "🌐 Server Components"
        SC1[WebSocket Server<br/>asyncio + websockets]
        SC2[Model Manager<br/>BERT-base + LoRA/Full]
        SC3[Aggregation Engine<br/>FedAvg Algorithm]
        SC4[Knowledge Distillation<br/>Teacher Inference]
    end
    
    subgraph "👥 Client Components"
        CC1[WebSocket Client<br/>Connection Handler]
        CC2[Data Loader<br/>GLUE Datasets]
        CC3[Model Trainer<br/>Local Training Loop]
        CC4[Parameter Manager<br/>Serialization/Deserialization]
    end
    
    subgraph "📊 Data Pipeline"
        DP1[SST-2 Dataset<br/>Sentiment Analysis]
        DP2[QQP Dataset<br/>Question Pairs]
        DP3[STS-B Dataset<br/>Similarity Scores]
    end
    
    subgraph "🔄 Communication Layer"
        CL1[JSON Serialization<br/>Parameter Transfer]
        CL2[WebSocket Protocol<br/>Real-time Streaming]
        CL3[Error Handling<br/>Robust Connections]
    end
    
    SC1 <--> CC1
    SC2 <--> CC3
    SC3 <--> CC4
    SC4 <--> CC3
    
    CC2 --> DP1
    CC2 --> DP2
    CC2 --> DP3
    
    CC1 <--> CL2
    CC4 <--> CL1
    SC1 <--> CL3
```

---

## 📈 Performance & Resource Architecture

### 💻 Resource Utilization Comparison

```mermaid
graph LR
    subgraph "📊 LoRA Approach"
        LR1[Memory Usage<br/>Low: LoRA gradients only]
        LR2[Network Traffic<br/>Light: 64KB per round]
        LR3[Training Time<br/>Fast: 2-3 minutes]
        LR4[Parameter Updates<br/>Efficient: 0.8% of model]
    end
    
    subgraph "⚡ No-LoRA Approach"
        NR1[Memory Usage<br/>High: Full gradients]
        NR2[Network Traffic<br/>Heavy: 18MB per round]
        NR3[Training Time<br/>Moderate: 4-6 minutes]
        NR4[Parameter Updates<br/>Complete: 100% of model]
    end
    
    subgraph "🎯 Trade-offs"
        T1[Efficiency vs Simplicity]
        T2[Speed vs Convergence]
        T3[Resources vs Performance]
    end
    
    LR1 --> T1
    NR1 --> T1
    LR3 --> T2
    NR3 --> T2
    LR2 --> T3
    NR2 --> T3
    
    style LR1 fill:#4caf50
    style LR2 fill:#4caf50
    style NR1 fill:#ff9800
    style NR2 fill:#ff9800
```

---

## 🔒 Security & Robustness Architecture

### 🛡️ Error Handling & Resilience

```mermaid
graph TD
    subgraph "🔐 Connection Security"
        S1[WebSocket Validation<br/>Client Authentication]
        S2[Parameter Validation<br/>Shape & Type Checking]
        S3[Timeout Handling<br/>Client Disconnection]
    end
    
    subgraph "🛠️ Error Recovery"
        E1[Connection Retry<br/>Automatic Reconnection]
        E2[Parameter Fallback<br/>Previous Round Weights]
        E3[Graceful Degradation<br/>Partial Client Sets]
    end
    
    subgraph "📊 Monitoring"
        M1[Training Metrics<br/>Loss & Accuracy Tracking]
        M2[System Health<br/>Resource Monitoring]
        M3[Client Status<br/>Connection State]
    end
    
    S1 --> E1
    S2 --> E2
    S3 --> E3
    E1 --> M3
    E2 --> M1
    E3 --> M2
    
    style S1 fill:#2196f3
    style E1 fill:#4caf50
    style M1 fill:#ff9800
```

---

## 🎯 Implementation Files Mapping

### 📁 Architecture to Code Mapping

| **Architectural Component** | **LoRA Implementation** | **No-LoRA Implementation** |
|----------------------------|-------------------------|----------------------------|
| **Main Script** | `fixed_streaming_glue.py` | `streaming_no_lora.py` |
| **Demo Runner** | `run_fixed_streaming.sh` | `run_no_lora_demo.sh` |
| **Server Class** | `FixedFederatedServer` | `NoLoRAFederatedServer` |
| **Client Class** | `FixedFederatedClient` | `NoLoRAFederatedClient` |
| **Model Class** | `GLUEModel` (with LoRA) | `NoLoRABERTModel` (full) |
| **Config Class** | `FixedGLUEConfig` | `NoLoRAConfig` |
| **Dataset Class** | `GLUEDataset` | `GLUEDataset` (same) |
| **KD Function** | `knowledge_distillation_loss` | `knowledge_distillation_loss` |

---

## 🚀 Deployment Architecture

### 🌐 Production Deployment Options

```mermaid
graph TB
    subgraph "☁️ Cloud Deployment"
        CD1[Server: AWS/GCP Instance<br/>GPU-enabled for teacher model]
        CD2[Clients: Edge Devices<br/>Mobile/IoT with local data]
        CD3[Load Balancer<br/>Multiple server instances]
    end
    
    subgraph "🏠 Local Deployment"
        LD1[Server: Local Machine<br/>High-memory for full training]
        LD2[Clients: Docker Containers<br/>Simulated federated setup]
        LD3[Network: Local WebSocket<br/>Fast communication]
    end
    
    subgraph "🔬 Research Setup"
        RD1[Jupyter Notebooks<br/>Interactive experiments]
        RD2[Automated Scripts<br/>Batch experiments]
        RD3[Metrics Collection<br/>Tensorboard/Wandb]
    end
    
    CD1 <--> CD2
    CD1 <--> CD3
    LD1 <--> LD2
    LD1 <--> LD3
    RD1 <--> RD2
    RD2 <--> RD3
    
    style CD1 fill:#e3f2fd
    style LD1 fill:#f3e5f5
    style RD1 fill:#fff3e0
```

---

## 🎓 Educational Architecture

### 📚 Learning Progression

```mermaid
graph LR
    subgraph "🎯 Beginner Level"
        B1[Basic Federated Learning<br/>Centralized aggregation]
        B2[Simple Knowledge Distillation<br/>Teacher-student setup]
    end
    
    subgraph "🚀 Intermediate Level"
        I1[Streaming Communication<br/>WebSocket implementation]
        I2[Multi-task Learning<br/>GLUE datasets]
        I3[Cross-architecture Transfer<br/>BERT variants]
    end
    
    subgraph "🏆 Advanced Level"
        A1[Parameter-Efficient Learning<br/>LoRA integration]
        A2[Production Deployment<br/>Scalable architecture]
        A3[Research Extensions<br/>Novel algorithms]
    end
    
    B1 --> I1
    B2 --> I2
    I1 --> A1
    I2 --> A2
    I3 --> A3
    
    style B1 fill:#c8e6c9
    style I1 fill:#fff9c4
    style A1 fill:#ffcdd2
```

---

## 🔍 Debugging & Development Architecture

### 🛠️ Development Workflow

```mermaid
graph TD
    subgraph "🔧 Development Phase"
        D1[Code Implementation<br/>Python + PyTorch]
        D2[Unit Testing<br/>Component validation]
        D3[Integration Testing<br/>End-to-end flow]
    end
    
    subgraph "🐛 Debugging Tools"
        DB1[Logging System<br/>Detailed execution traces]
        DB2[Parameter Inspection<br/>Weight visualization]
        DB3[Connection Monitoring<br/>WebSocket debugging]
    end
    
    subgraph "📊 Performance Analysis"
        PA1[Training Metrics<br/>Loss & accuracy curves]
        PA2[Resource Usage<br/>Memory & compute profiling]
        PA3[Network Analysis<br/>Bandwidth utilization]
    end
    
    D1 --> D2
    D2 --> D3
    D3 --> DB1
    DB1 --> DB2
    DB2 --> DB3
    DB3 --> PA1
    PA1 --> PA2
    PA2 --> PA3
    
    style D1 fill:#e8f5e8
    style DB1 fill:#fff3e0
    style PA1 fill:#f3e5f5
```

---

## 🎉 Summary: Architecture Benefits

### ✅ Key Architectural Advantages

| **Aspect** | **LoRA Architecture** | **No-LoRA Architecture** |
|------------|----------------------|--------------------------|
| **🚀 Efficiency** | Parameter-efficient, low memory | Traditional, high memory |
| **🔄 Flexibility** | Modular adapters, easy swapping | Full control, direct training |
| **📊 Scalability** | Scales to many clients easily | Better for fewer, powerful clients |
| **🎓 Learning** | Modern PEFT techniques | Classical federated learning |
| **🔬 Research** | Cutting-edge parameter efficiency | Pure knowledge distillation study |

### 🌟 Both Architectures Provide:
- ✅ **Real-time streaming** via WebSocket communication
- ✅ **Cross-architecture learning** between BERT variants  
- ✅ **Multi-task federated learning** across GLUE tasks
- ✅ **Production-ready** error handling and scalability
- ✅ **Educational value** for federated learning concepts
- ✅ **Research extensibility** for novel algorithms

---

*This architecture document provides a comprehensive view of both streaming federated learning approaches, enabling informed decisions about which implementation best fits your specific use case, resources, and research goals.* 🚀
