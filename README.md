# 🪖 Helmet Compliance Monitoring System

**Automated Safety Compliance Through Computer Vision**

Safety rule violations, like not wearing helmets, are hard to track manually. A computer vision system can automate helmet compliance monitoring, enhance workplace safety, and reduce accident severity without needing extra staff.

![Helmet Detection Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

## ⚠️ IMPORTANT LEGAL NOTICE

**🚨 THIS PROJECT IS PROTECTED BY COPYRIGHT LAW 🚨**

This repository is monitored for unauthorized copying, forking, or distribution. All activities are tracked and logged.

---

## 🎯 Problem Statement

Traditional manual safety monitoring faces several challenges:
- **Human Error**: Manual oversight can miss violations
- **Resource Intensive**: Requires dedicated safety personnel
- **Inconsistent Monitoring**: Cannot provide 24/7 surveillance
- **Delayed Response**: Violations detected after incidents occur
- **Documentation Issues**: Difficult to maintain compliance records

## 💡 Solution

Our AI-powered system provides:
- **Real-time Detection**: Instant helmet compliance verification
- **Automated Monitoring**: Continuous surveillance without human intervention
- **Accurate Documentation**: Automated violation logging and reporting
- **Cost-effective**: Reduces need for additional safety staff
- **Scalable**: Can monitor multiple locations simultaneously

## 🚀 Key Features

### 🔍 Detection Capabilities
- **Real-time Helmet Detection** - Instant compliance verification
- **Live Camera Integration** - Continuous monitoring through webcam/IP cameras
- **Batch Image Processing** - Analyze multiple images simultaneously
- **High Accuracy** - Advanced AI model with adjustable confidence thresholds

### 📊 Monitoring & Analytics
- **Violation Logging** - Automatic incident recording with timestamps
- **Compliance Reporting** - Generate detailed safety compliance reports
- **Statistics Dashboard** - Track compliance rates and trends
- **Export Functionality** - Download reports in CSV format

### ⚙️ User-Friendly Interface
- **Intuitive Web Interface** - Easy-to-use Streamlit dashboard
- **Adjustable Thresholds** - Customize detection sensitivity
- **Multiple Input Methods** - Upload images, use camera, or batch process
- **Real-time Feedback** - Instant safety status notifications

## 🏭 Use Cases

### Industrial Applications
- **Construction Sites** - Monitor workers in hard hat zones
- **Manufacturing Plants** - Ensure safety compliance in production areas
- **Warehouses** - Automated safety checks in material handling zones
- **Mining Operations** - Critical safety monitoring in hazardous environments

### Benefits for Organizations
- **Reduced Accidents** - Proactive safety violation prevention
- **Lower Insurance Costs** - Improved safety records
- **Regulatory Compliance** - Meet OSHA and safety standards
- **Enhanced Productivity** - Automated monitoring frees staff for other tasks

## 📋 Technical Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space
- **Camera**: Optional (for live detection)

### Dependencies
```
streamlit==1.28.0
tensorflow==2.13.0
opencv-python==4.8.0.74
Pillow==10.0.0
numpy==1.24.3
pandas==2.0.3
```

## 🔧 Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/HarshitBhalani/Helmet-Compliance-Monitoring.git
cd Helmet-Compliance-Monitoring
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv helmet_env
helmet_env\Scripts\activate

# macOS/Linux
python -m venv helmet_env
source helmet_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Setup
**⚠️ Important**: Model files are not included in this repository due to size constraints.

**Option A: Train Your Own Model**
1. Visit [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Create image classification project
3. Upload helmet/no-helmet training images
4. Train and export as TensorFlow model
5. Download and rename to `model.h5`
6. Place in project root directory

**Option B: Use Pre-trained Model** (If Available)
1. Contact repository maintainer for model file
2. Place `model.h5` in project root directory

### 5. Run Application
```bash
streamlit run app.py
```

### 6. Access Interface
Open your browser and navigate to: `http://localhost:8501`

## 📱 How to Use

### 1. Single Image Detection
- Select "📷 Image Upload" mode
- Upload an image file (JPG, PNG, BMP)
- View detection results and confidence scores
- System automatically logs violations

### 2. Live Camera Monitoring
- Select "📹 Live Camera" mode  
- Allow camera permissions
- Capture photos for real-time analysis
- Get instant safety compliance feedback

### 3. Batch Processing
- Select "📁 Batch Processing" mode
- Upload multiple images at once
- Generate comprehensive compliance report
- Export results for documentation

### 4. View Reports
- Select "📊 Violation Logs" mode
- Review all detected violations
- Analyze compliance trends
- Export data for regulatory reporting

## 🎛️ Configuration Options

### Detection Threshold Settings
- **0.5-0.6**: Lenient (reduces false violations, may miss some cases)
- **0.7**: Balanced (recommended for most environments)
- **0.8-0.9**: Strict (ideal for high-risk environments)

### Customization Tips
- Adjust threshold based on your safety requirements
- Higher thresholds for critical safety zones
- Lower thresholds for general monitoring areas

## 🤖 AI Model Information

### Model Specifications
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Classes**: 2 (With Helmet, Without Helmet)
- **Framework**: TensorFlow/Keras
- **Training**: Supervised learning on labeled helmet images

### Performance Metrics
- **Accuracy**: Depends on training data quality
- **Processing Speed**: Real-time capability
- **Memory Usage**: Optimized for standard hardware

## 📊 Sample Output

### Compliance Detection
```
✅ HELMET DETECTED - COMPLIANT
Status: SAFE ✓
Confidence Level: 89.2%
```

### Violation Alert
```
❌ NO HELMET DETECTED - VIOLATION
Status: UNSAFE ⚠️
Confidence Level: 94.7%

🚨 SAFETY VIOLATION ALERT
Immediate Actions Required:
- 🛑 Stop work immediately  
- 🪖 Provide safety helmet
- 📋 Brief worker on safety protocols
- 📝 Document the incident
```

## 🏗️ Project Structure

```
helmet_streamlit_app/
├── model/                   # Model directory
│   ├── model.json          # Model architecture in JSON format
│   ├── metadata.json       # Model metadata and configuration
│   └── weights.bin         # Model weights in binary format
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── labels.txt             # Class labels for model
├── README.md              # Project documentation
├── .gitignore            # Git exclusion rules
└── model.h5              # Complete Keras/TensorFlow model file
```

## 🔒 Security & Privacy

- **Local Processing**: All detection happens on your local machine
- **No Data Upload**: Images are not sent to external servers
- **Privacy Compliant**: Suitable for sensitive workplace environments
- **Secure Logging**: Violation logs stored locally

## 🚀 Deployment Options

### Local Deployment
- Run on local machine for single-user access
- Ideal for testing and small-scale monitoring

### Network Deployment
- Deploy on internal server for multi-user access
- Access from multiple devices on same network

### Cloud Deployment (Advanced)
- Deploy on Streamlit Cloud, Heroku, or AWS
- Requires model hosting solution (Google Drive, etc.)

## 🛠️ Troubleshooting

### Common Issues

**Model Loading Error**
```
❌ model.h5 file not found!
```
**Solution**: Ensure model.h5 is in the project root directory

**Camera Access Error**
```
Permission denied for camera access
```
**Solution**: Grant camera permissions in browser settings

**Low Detection Accuracy**
```
Many false positives/negatives
```
**Solution**: Adjust detection threshold or retrain model with better data

### Performance Optimization
- Use GPU-enabled TensorFlow for faster processing
- Optimize image resolution for speed vs accuracy balance
- Consider model quantization for mobile deployment

## 🤝 Contributing

We welcome contributions to improve the system!

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- Model accuracy improvements
- Additional safety equipment detection
- Mobile app development
- Integration with IoT devices
- Advanced analytics features

---

## 📄 LICENSE - ALL RIGHTS RESERVED

### 🚨 PROPRIETARY SOFTWARE - STRICT COPYRIGHT PROTECTION 🚨

**© 2025 Harshit Bhalani. All Rights Reserved.**

This project is proprietary and confidential software. All rights are reserved by the copyright holder.

### ⚖️ LEGAL TERMS & CONDITIONS

**UNAUTHORIZED USE PROHIBITED**

This repository and its contents are protected by copyright law and international treaties. Any unauthorized copying, distribution, modification, or use of this software is **STRICTLY PROHIBITED** and will result in legal action.

### 🚫 WHAT IS NOT PERMITTED:

- ❌ **Copying** any portion of this code
- ❌ **Forking** or cloning this repository for personal use
- ❌ **Redistributing** in any form (source code, compiled, or modified)
- ❌ **Commercial use** without explicit written permission
- ❌ **Creating derivative works** based on this project
- ❌ **Reverse engineering** or decompiling
- ❌ **Removing** copyright notices or attribution
- ❌ **Using** for competing products or services

### ✅ WHAT IS PERMITTED:

- ✅ **Viewing** the code for educational purposes only
- ✅ **Learning** from the implementation concepts
- ✅ **Discussing** the project in academic contexts
- ✅ **Linking** to this repository (not copying)

### 🔍 MONITORING & ENFORCEMENT

**This repository is actively monitored for copyright violations.**

- All forks, downloads, and access attempts are logged
- Automated detection systems identify unauthorized use
- Legal action will be taken against violators
- DMCA takedown notices will be issued for violations

### ⚠️ VIOLATION CONSEQUENCES

Unauthorized use of this software may result in:
- **Immediate legal action** under copyright law
- **Cease and desist** orders
- **Financial penalties** and damages
- **Criminal prosecution** under applicable laws

### 📧 LICENSING INQUIRIES

For licensing, permission, or commercial use inquiries, contact:
- **Email**: harshitbhalani187@gmail.com
- **GitHub**: @HarshitBhalani
- **Subject**: "Helmet Detection System - Licensing Inquiry"

### 🛡️ DISCLAIMER

This software is provided "as is" without warranty of any kind. The author shall not be liable for any damages arising from the use of this software.

---

## 📞 Support & Contact

### Getting Help
- **Issues**: Report bugs via [GitHub Issues](https://github.com/HarshitBhalani/Helmet-Compliance-Monitoring/issues)
- **Discussions**: Join project discussions
- **Email**: [harshitbhalani187@gmail.com]

### FAQ

**Q: Can this detect other safety equipment?**
A: Currently focused on helmets, but can be extended for vests, gloves, etc.

**Q: What accuracy can I expect?**
A: Depends on training data quality, typically 85-95% with good data.

**Q: Can it work with IP cameras?**
A: Yes, with minor code modifications for RTSP streams.

**Q: Is it suitable for outdoor use?**
A: Yes, but performance may vary with lighting conditions.

## 🙏 Acknowledgments

- **TensorFlow Team** for the amazing ML framework
- **Streamlit** for the intuitive web app framework  
- **OpenCV** for computer vision capabilities
- **Google Teachable Machine** for accessible model training

## 📈 Future Enhancements

- [ ] Multi-person detection in single image
- [ ] Integration with existing security systems
- [ ] Mobile application development
- [ ] Advanced analytics and predictive insights
- [ ] Support for multiple safety equipment types
- [ ] Real-time video stream processing
- [ ] Database integration for enterprise use
- [ ] API development for third-party integrations

---

<div align="center">

**⚡ Automated Safety Monitoring for a Safer Workplace ⚡**

**🔒 PROTECTED BY COPYRIGHT LAW 🔒**

Made with ❤️ for workplace safety by Harshit Bhalani

**© 2025 All Rights Reserved**

</div>

---

### 🚨 FINAL WARNING

**This project is protected by copyright law. Any unauthorized copying, distribution, or use will result in immediate legal action. We actively monitor and track all repository activity.**
