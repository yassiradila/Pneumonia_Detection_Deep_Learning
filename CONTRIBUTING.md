# Contributing to Pneumonia Detection (InceptionV3)

Thank you for your interest in contributing to this project! We welcome improvements to the deep learning pipelines, documentation, evaluation benchmarks, and deployment tooling.

## How to Contribute

1. **Fork the Repository**:
   Click the "Fork" button at the top right of this repository page.

2. **Clone your Fork**:
   ```bash
   git clone https://github.com/yassiradila/Pneumonia_Detection_Deep_Learning.git
   cd Pneumonia_Detection_Deep_Learning
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/awesome-improvement
   ```

4. **Install Dependencies in a Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

5. **Run the Test Suite**:
   ```bash
   python -m unittest discover tests
   ```

6. **Commit & Push**:
   Keep commit messages descriptive and focused.
   ```bash
   git commit -m "feat: add Grad-CAM explainability visualization"
   git push origin feature/awesome-improvement
   ```

7. **Open a Pull Request**:
   Submit your pull request describing the changes and referencing any related issues.

## Reporting Issues
Please open a GitHub issue if you encounter bugs, broken paths, or reproducibility discrepancies. Include your Python version, TensorFlow version, and GPU/CPU environment details.
