# 🎯 ArcFace PyTorch Implementation

[![GitHub stars](https://img.shields.io/github/stars/KELLO83/Face-Recognition?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/KELLO83/Face-Recognition?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition/network)
[![GitHub license](https://img.shields.io/github/license/KELLO83/Face-Recognition?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition/blob/main/LICENSE)

> **High-Performance Face Recognition System with ArcFace Loss Implementation**

## 🏆 Model Performance Dashboard

<div align="center">

### Top Performers

[![Accuracy Leader](https://img.shields.io/badge/🏆_Accuracy_Leader-AdaFace_ir101-gold?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition)
[![Speed Leader](https://img.shields.io/badge/⚡_Speed_Leader-ArcFace_irsnet50-blue?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition)
[![Efficiency Leader](https://img.shields.io/badge/💾_Efficiency_Leader-ArcFace_irsnet50-green?style=for-the-badge)](https://github.com/KELLO83/Face-Recognition)

</div>

## 📊 Model Performance Comparison

### Performance Summary Table

| Model | TAR@FAR 0.01% | Rank-1 Acc | Latency (ms) | Throughput (FPS) | Size (MB) | GFLOPs |
|-------|---------------|-------------|--------------|------------------|-----------|---------|
| **AdaFace ir101 (webface12m)** ⭐| **96.08%** | **99.19%** | 13.57 | 73.67 | 483.74 | 12.122 |
| TopoFR R200 (Glint360k) | 95.39% | 98.95% | 45.74 | 21.86 | 672.28 | 23.485 |
| AdaFace ir101 (webface4m) | 92.64% | 99.06% | 13.57 | 73.67 | 483.74 | 12.122 |
| TopoFR R100 (Glint360k) | 92.34% | 98.79% | 24.62 | 40.62 | 373.88 | 12.128 |
| TopoFR R200 (ms1mv2) | 89.19% | 98.23% | 35.55 | 28.13 | 672.28 | 23.485 |
| TransFace vit_l (Glint360K) | 88.43% | 98.53% | 16.93 | 59.05 | 1226.04 | 24.621 |
| AdaFace ir101 (MS1MV3) | 88.01% | 98.63% | 13.57 | 73.67 | 483.74 | 12.122 |
| ArcFace resnet100 (ms1mv2) | 81.81% | 97.64% | 11.46 | 87.26 | 486.98 | 12.128 |
| TransFace vit_s (Glint360K) | 75.31% | 96.42% | 8.48 | 117.92 | 393.70 | 5.513 |
| AdaFace resnet50 (casia) | 50.35% | 89.18% | 6.93 | 144.28 | 315.59 | 6.325 |
| **ArcFace irsnet50 (casia)** ⚡ | 41.57% | 91.74% | **5.45** | **183.58** | **292.27** | 6.319 |

### 📈 Performance Highlights

```
                    Accuracy  Speed   Size   Overall
AdaFace ir101       ████████  ██████  ██████ ████████
TopoFR R200         ████████  ██      ████   ██████ 
ArcFace irsnet50    ████      ████████ ██████ ██████
TransFace vit_l     ███████   █████   █      █████
```

### Performance Analysis

🏆 **Best Accuracy**: AdaFace ir101 (webface12m) - 96.08% TAR@FAR 0.01%  
⚡ **Fastest Model**: ArcFace irsnet50 - 5.45ms latency, 183.58 FPS  
💾 **Smallest Model**: ArcFace irsnet50 - 292.27 MB  
⚖️ **Best Balance**: AdaFace ir101 (webface12m) - High accuracy + Reasonable speed

### Interactive Chart
> **[📊 View Interactive Performance Chart](https://kello83.github.io/Face-Recognition/model_performence.html)**

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/KELLO83/Face-Recognition.git
cd Face-Recognition
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python train.py

# Docker training
docker build -t arcface-pytorch .
docker run --gpus all -v $(pwd)/data:/workspace/data arcface-pytorch
```

## 📁 Project Structure

```
arcface-pytorch/
├── 📊 model_performence.html     # Interactive performance chart
├── 🐳 Dockerfile               # Docker configuration
├── 🏃 train.py                 # Main training script
├── ⚙️ config/                  # Configuration files
├── 📊 data/                    # Dataset handling
├── 🧠 models/                  # Model architectures
├── 🛠️ utils/                   # Utility functions
├── 📈 logs/                    # Training logs
├── 💾 checkpoints/             # Model checkpoints
└── 📋 requirements.txt         # Dependencies
```

## 🔧 Features

- ✅ **ArcFace Loss Implementation**
- ✅ **Multiple Backbone Architectures**
- ✅ **Face Detection Integration (YOLO)**
- ✅ **Data Augmentation Pipeline**
- ✅ **Early Stopping & Scheduling**
- ✅ **Docker Support**
- ✅ **Mixed Precision Training**
- ✅ **Real-time Performance Monitoring**

## 📊 Detailed Performance Data

| Model | Backbone | ROC-AUC | EER | Accuracy | TAR@FAR 0.01% | Rank-1 Acc | Rank-5 Acc | Latency (ms) | Throughput (FPS) | Size (MB) | GFLOPs | Mparams | TAR@FAR 1% | TAR@FAR 0.1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TransFace vit_l (Glint360K) | VIT · Glint360K | 0.9995 | 0.0088 | 0.9912 | 0.8843 | 0.9853 | 0.9989 | 16.93 | 59.05 | 1226.04 | 24.621 | 271.630 | 0.9921 | 0.9615 |
| TransFace vit_s (Glint360K) | VIT · Glint360K | 0.9980 | 0.0196 | 0.9804 | 0.7531 | 0.9642 | 0.9928 | 8.48 | 117.92 | 393.70 | 5.513 | 86.661 | 0.9676 | 0.8897 |
| TopoFR R200 (Glint360k) | ResNet-200 · Glint360K | 0.9999 | 0.0037 | 0.9963 | 0.9539 | 0.9895 | 0.9997 | 45.74 | 21.86 | 672.28 | 23.485 | 119.096 | 0.9985 | 0.9897 |
| TopoFR R200 (ms1mv2) | ResNet-200 · MS1MV2 | 0.9996 | 0.0083 | 0.9916 | 0.8919 | 0.9823 | 0.9986 | 35.55 | 28.13 | 672.28 | 23.485 | 119.096 | 0.9929 | 0.9634 |
| TopoFR R100 (Glint360k) | ResNet-100 · Glint360K | 0.9998 | 0.0058 | 0.9942 | 0.9234 | 0.9879 | 0.9994 | 24.62 | 40.62 | 373.88 | 12.128 | 65.418 | 0.9966 | 0.9777 |
| AdaFace ir101 (webface12m) | IR-101 · WebFace12M | 0.9999 | 0.0030 | 0.9970 | 0.9608 | 0.9919 | 1.0000 | 13.57 | 73.67 | 483.74 | 12.122 | 65.151 | 0.9992 | 0.9921 |
| AdaFace ir101 (webface4m) | IR-101 · WebFace4M | 0.9998 | 0.0052 | 0.9948 | 0.9264 | 0.9906 | 0.9999 | 13.57 | 73.67 | 483.74 | 12.122 | 65.151 | 0.9973 | 0.9814 |
| AdaFace ir101 (MS1MV3) | IR-101 · MS1MV3 | 0.9995 | 0.0093 | 0.9907 | 0.8801 | 0.9863 | 0.9991 | 13.57 | 73.67 | 483.74 | 12.122 | 65.151 | 0.9913 | 0.9591 |
| AdaFace resnet50 (casia) | ResNet-50 · Casia | 0.9873 | 0.0578 | 0.9422 | 0.5035 | 0.8918 | 0.9653 | 6.93 | 144.28 | 315.59 | 6.325 | 43.586 | 0.8375 | 0.6738 |
| ArcFace resnet100 (ms1mv2) | ResNet-100 · MS1M | 0.9988 | 0.0148 | 0.9852 | 0.8181 | 0.9764 | 0.9964 | 11.46 | 87.26 | 486.98 | 12.128 | 65.156 | 0.9795 | 0.9216 |
| ArcFace irsnet50 (casia) | ResNet-50 · MS1M | 0.9896 | 0.0489 | 0.9511 | 0.4157 | 0.9174 | 0.9718 | 5.45 | 183.58 | 292.27 | 6.319 | 43.575 | 0.8524 | 0.6373 |
  
                ];
                
                // flag에 따른 데이터 매핑
                const dataMapping = {
                    'TAR@FAR 0.01%': {
                        data: jsonData.map(row => row["TAR@FAR 0.01%"]),
                        label: 'TAR @ FAR 0.01%',
                        title: 'TAR @ FAR 0.01% Comparison',
                        yAxisTitle: 'TAR @ FAR 0.01%',
                        unit: '%',
                        multiplier: 100,
                        maxValue: 1.0
                    },
                    'Rank-1 Acc': {
                        data: jsonData.map(row => row["Rank-1 Acc"]),
                        label: 'Rank-1 Accuracy',
                        title: 'Rank-1 Accuracy Comparison',
                        yAxisTitle: 'Rank-1 Accuracy',
                        unit: '%',
                        multiplier: 100,
                        maxValue: 1.0
                    },
                    'Rank-5 Acc': {
                        data: jsonData.map(row => row["Rank-5 Acc"]),
                        label: 'Rank-5 Accuracy',
                        title: 'Rank-5 Accuracy Comparison',
                        yAxisTitle: 'Rank-5 Accuracy',
                        unit: '%',
                        multiplier: 100,
                        maxValue: 1.0
                    },
                    'Average Latency (ms)': {
                        data: jsonData.map(row => row["Average Latency (ms)"]),
                        label: 'Average Latency',
                        title: 'Average Latency Comparison',
                        yAxisTitle: 'Latency (ms)',
                        unit: 'ms',
                        multiplier: 1,
                        maxValue: null
                    },
                    'Throughput (FPS)': {
                        data: jsonData.map(row => row["Throughput (FPS)"]),
                        label: 'Throughput',
                        title: 'Throughput Comparison',
                        yAxisTitle: 'Throughput (FPS)',
                        unit: ' FPS',
                        multiplier: 1,
                        maxValue: null
                    },
                    'Estimated Total Size(MB)': {
                        data: jsonData.map(row => row["Estimated Total Size(MB)"]),
                        label: 'Model Size',
                        title: 'Model Size Comparison',
                        yAxisTitle: 'Size (MB)',
                        unit: ' MB',
                        multiplier: 1,
                        maxValue: null
                    },
                    'GFLOPs': {
                        data: jsonData.map(row => row["GFLOPs"]),
                        label: 'GFLOPs',
                        title: 'GFLOPs Comparison',
                        yAxisTitle: 'GFLOPs',
                        unit: ' GFLOPs',
                        multiplier: 1,
                        maxValue: null
                    },
                    'Mparams': {
                        data: jsonData.map(row => row["Mparams"]),
                        label: 'Mparams',
                        title: 'Model Parameters Comparison',
                        yAxisTitle: 'Parameters (M)',
                        unit: ' M',
                        multiplier: 1,
                        maxValue: null
                    },
                    'EER': {
                        data: jsonData.map(row => row["EER"]),
                        label: 'EER (Equal Error Rate)',
                        title: 'EER (Equal Error Rate) Comparison',
                        yAxisTitle: 'EER',
                        unit: '%',
                        multiplier: 100,
                        maxValue: null
                    }
                };

                const currentConfig = dataMapping[flag];
                if (!currentConfig) {
                    throw new Error(`Invalid flag: ${flag}`);
                }

                const labels = jsonData.map(row => row["모델명"]);

                const config = {
                  "type": "bar",
                  "data": {
                    "labels": labels,
                    "datasets": [
                      {
                        "label": "TAR @ FAR 0.01%",
                        "data": jsonData.map(row => row["TAR@FAR 0.01%"]),
                        "backgroundColor": "rgba(255, 99, 132, 0.8)",
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Rank-1 Accuracy",
                        "data": jsonData.map(row => row["Rank-1 Acc"]),
                        "backgroundColor": "rgba(255, 159, 64, 0.8)",
                        "borderColor": "rgba(255, 159, 64, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Rank-5 Accuracy",
                        "data": jsonData.map(row => row["Rank-5 Acc"]),
                        "backgroundColor": "rgba(255, 205, 86, 0.8)",
                        "borderColor": "rgba(255, 205, 86, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Average Latency (Normalized)",
                        "data": jsonData.map(row => row["Average Latency (ms)"] / 100), 
                        "backgroundColor": "rgba(75, 192, 192, 0.8)",
                        "borderColor": "rgba(75, 192, 192, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Throughput (Normalized)",
                        "data": jsonData.map(row => row["Throughput (FPS)"] / 200),
                        "backgroundColor": "rgba(54, 162, 235, 0.8)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Model Size (Normalized)",
                        "data": jsonData.map(row => row["Estimated Total Size(MB)"] / 1500), 
                        "backgroundColor": "rgba(153, 102, 255, 0.8)",
                        "borderColor": "rgba(153, 102, 255, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "GFLOPs (Normalized)",
                        "data": jsonData.map(row => row["GFLOPs"] / 30), 
                        "backgroundColor": "rgba(255, 193, 7, 0.8)",
                        "borderColor": "rgba(255, 193, 7, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "Mparams (Normalized)",
                        "data": jsonData.map(row => row["Mparams"] / 300), 
                        "backgroundColor": "rgba(156, 39, 176, 0.8)",
                        "borderColor": "rgba(156, 39, 176, 1)",
                        "borderWidth": 2
                      },
                      {
                        "label": "EER (Normalized)",
                        "data": jsonData.map(row => row["EER"] / 0.1), 
                        "backgroundColor": "rgba(255, 255, 255, 0.8)",
                        "borderColor": "rgba(255, 255, 255, 1)",
                        "borderWidth": 2
                      }
                    ]
                  },
                  "options": {
                    "responsive": true,
                    "maintainAspectRatio": false,
                    "scales": {
                      "y": {
                        "beginAtZero": true,
                        "max": 1.0,
                        "title": {
                          "display": true,
                          "text": "Normalized Values (0-1)",
                          "color": "#ffffff"
                        },
                        "ticks": {
                          "color": "#ffffff",
                          "callback": function(value) {
                            return (value * 100).toFixed(0) + '%';
                          }
                        },
                        "grid": {
                          "color": "rgba(255, 255, 255, 0.2)"
                        }
                      },
                      "x": {
                        "title": {
                          "display": true,
                          "text": "Model",
                          "color": "#ffffff"
                        },
                        "ticks": {
                          "color": "#ffffff",
                          "maxRotation": 0,
                          "minRotation": 0,
                          "font": {
                            "size": 14, // x축 모델 이름 라벨 크기
                            "color": "#ffffff"
                          }
                        },
                        "grid": {
                          "color": "rgba(255, 255, 255, 0.2)"
                        }
                      }
                    },
                    "plugins": {
                      "legend": {
                        "labels": {
                          "color": "#ffffff",
                          "font": {
                            "color": "#ffffff"
                          },
                          "generateLabels": function(chart) {
                            const original = Chart.defaults.plugins.legend.labels.generateLabels;
                            const labels = original.call(this, chart);
                            return labels;
                          }
                        }
                      },
                      "title": {
                        "display": true,
                        "text": "Model Performance Comparison (All Metrics)",
                        "color": "#ffffff",
                        "font": {
                            "size": 30
                        }
                      },
                      "tooltip": {
                        "callbacks": {
                          "label": function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                              label += ': ';
                            }
                            
                            const datasetIndex = context.datasetIndex;
                            const dataIndex = context.dataIndex;
                            let originalValue;
                            
                            if (datasetIndex === 0) { // TAR@FAR 0.01%
                              originalValue = (context.parsed.y * 100).toFixed(1) + '%';
                            } else if (datasetIndex === 1 || datasetIndex === 2) { // Rank-1, Rank-5 Acc
                              originalValue = (context.parsed.y * 100).toFixed(1) + '%';
                            } else if (datasetIndex === 3) { // Average Latency
                              originalValue = (context.parsed.y * 100).toFixed(1) + 'ms';
                            } else if (datasetIndex === 4) { // Throughput
                              originalValue = (context.parsed.y * 200).toFixed(1) + ' FPS';
                            } else if (datasetIndex === 5) { // Model Size
                              originalValue = (context.parsed.y * 1500).toFixed(0) + ' MB';
                            } else if (datasetIndex === 6) { // GFLOPs
                              originalValue = (context.parsed.y * 30).toFixed(2) + ' GFLOPs';
                            } else if (datasetIndex === 7) { // Mparams
                              originalValue = (context.parsed.y * 300).toFixed(1) + ' M';
                            } else if (datasetIndex === 8) { // EER
                              originalValue = (context.parsed.y * 0.1 * 100).toFixed(2) + '%';
                            }
                            
                            label += originalValue;
                            return label;
                          }
                        }
                      }
                    }
                  }
                };

                const ctx = document.getElementById('myChart').getContext('2d');
                new Chart(ctx, config);
                
            } catch (error) {
                console.error('Error creating chart:', error);
                const chartContainer = document.querySelector('.chart-container');
                chartContainer.innerHTML = `<p style="color: red; text-align: center;">Error loading chart data: ${error.message}</p>`;
            }
        }

        createChart();
    </script>
</body>
</html>