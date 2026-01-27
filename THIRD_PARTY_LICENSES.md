# Third-Party Licenses

This project references and uses various open-source projects. Below are the licenses for all third-party repositories and libraries mentioned in this documentation.

---

## Table of Contents

1. [Edge Deployment Projects](#edge-deployment)
2. [Server-Side GPU Projects](#server-side-gpu)
3. [Python Libraries](#python-libraries)
4. [Research Papers and Academic Work](#academic-work)

---

## Edge Deployment Projects {#edge-deployment}

### 1. WORLD Vocoder

**Repository**: https://github.com/mmorise/World

**License**: Modified BSD License (3-Clause BSD)

**Copyright**: Copyright 2012 Masanori Morise

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
```

**Citation**:
```
Morise, M., Yokomori, F., & Ozawa, K. (2016).
WORLD: A vocoder-based high-quality speech synthesis system for real-time applications.
IEICE Transactions on Information and Systems, 99(7), 1877-1884.
```

---

### 2. PyWorld (Python Wrapper for WORLD)

**Repository**: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

**License**: MIT License

**Copyright**: Copyright (c) 2017 Jeremy Hsu

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

### 3. Voice Gender Changer (PSOLA)

**Repository**: https://github.com/radinshayanfar/voice-gender-changer

**License**: MIT License

**Copyright**: Copyright (c) 2020 Radin Shayanfar

---

### 4. TinyVC

**Repository**: https://github.com/uthree/tinyvc

**License**: MIT License

**Copyright**: Copyright (c) 2023 uthree

---

### 5. Intel Neural Compressor

**Repository**: https://github.com/intel/neural-compressor

**License**: Apache License 2.0

**Copyright**: Copyright (c) 2021 Intel Corporation

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
```

---

### 6. ONNX Runtime

**Repository**: https://github.com/microsoft/onnxruntime

**License**: MIT License

**Copyright**: Copyright (c) Microsoft Corporation

---

## Server-Side GPU Projects {#server-side-gpu}

### 1. GPT-SoVITS

**Repository**: https://github.com/RVC-Boss/GPT-SoVITS

**License**: MIT License

**Copyright**: Copyright (c) 2024 RVC-Boss

**Note**: This project is for research and educational purposes. Commercial use requires careful consideration of voice cloning ethics and regulations.

---

### 2. RVC (Retrieval-based Voice Conversion)

**Repository**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

**License**: MIT License

**Copyright**: Copyright (c) 2023 RVC-Project

**Important Notice**: This project is intended for academic research and personal non-commercial use only. Any commercial use or voice cloning without consent is prohibited.

---

### 3. SoftVC VITS

**Repository**: https://github.com/svc-develop-team/so-vits-svc

**License**: AGPL-3.0 License (GNU Affero General Public License v3.0)

**Copyright**: Copyright (c) 2023 SVC Develop Team

**Key Points**:
- Source code modifications must be disclosed
- Network use is distribution (AGPL requirement)
- Commercial use allowed but with disclosure obligations
- Copyleft license (derivatives must use same license)

```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

---

### 4. Seed-VC

**Repository**: https://github.com/Plachtaa/seed-vc

**License**: MIT License

**Copyright**: Copyright (c) 2024 Plachtaa

---

### 5. FreeVC

**Repository**: https://github.com/OlaWod/FreeVC

**License**: MIT License

**Copyright**: Copyright (c) 2022 OlaWod

**Citation**:
```
Li, Y., et al. (2022).
FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion.
arXiv preprint arXiv:2210.15418.
```

---

### 6. DDSP-SVC

**Repository**: https://github.com/yxlllc/DDSP-SVC

**License**: MIT License

**Copyright**: Copyright (c) 2022 yxlllc

**Citation**:
```
DDSP-SVC: Differentiable Digital Signal Processing for Singing Voice Conversion
```

---

### 7. kNN-VC

**Repository**: https://github.com/bshall/knn-vc

**License**: MIT License

**Copyright**: Copyright (c) 2022 Benjamin van Niekerk

**Citation**:
```
van Niekerk, B., & Baas, M. (2023).
kNN-VC: Any-to-Any Voice Conversion with k-Nearest Neighbors.
Interspeech 2023.
```

---

### 8. VITS (Conditional Variational Autoencoder)

**Repository**: https://github.com/jaywalnut310/vits

**License**: MIT License

**Copyright**: Copyright (c) 2021 Jaehyeon Kim

**Citation**:
```
Kim, J., Kong, J., & Son, J. (2021).
Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.
ICML 2021.
```

---

### 9. HiFi-GAN (Vocoder)

**Repository**: https://github.com/jik876/hifi-gan

**License**: MIT License

**Copyright**: Copyright (c) 2020 Jungil Kong

**Citation**:
```
Kong, J., Kim, J., & Bae, J. (2020).
HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.
NeurIPS 2020.
```

---

## Python Libraries {#python-libraries}

### Core Dependencies

#### PyTorch

**Website**: https://pytorch.org/

**License**: BSD-3-Clause License

**Copyright**: Copyright (c) 2016-2024 Facebook, Inc. (Meta Platforms, Inc.)

---

#### NumPy

**Website**: https://numpy.org/

**License**: BSD 3-Clause License

**Copyright**: Copyright (c) 2005-2024, NumPy Developers

---

#### Librosa

**Repository**: https://github.com/librosa/librosa

**License**: ISC License

**Copyright**: Copyright (c) 2013-2024, librosa development team

```
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.
```

---

#### SoundFile

**Repository**: https://github.com/bastibe/python-soundfile

**License**: BSD 3-Clause License

**Copyright**: Copyright (c) 2013-2024, Bastian Bechtold

---

#### SciPy

**Website**: https://scipy.org/

**License**: BSD 3-Clause License

**Copyright**: Copyright (c) 2001-2024 SciPy Developers

---

#### TensorFlow / TensorFlow Lite

**Website**: https://www.tensorflow.org/

**License**: Apache License 2.0

**Copyright**: Copyright 2015 The TensorFlow Authors

---

#### ONNX

**Repository**: https://github.com/onnx/onnx

**License**: Apache License 2.0

**Copyright**: Copyright (c) ONNX Project Contributors

---

## Research Papers and Academic Work {#academic-work}

### Attribution Requirements

When using this project for academic or research purposes, please cite the relevant papers:

#### WORLD Vocoder
```
@article{morise2016world,
  title={WORLD: a vocoder-based high-quality speech synthesis system for real-time applications},
  author={Morise, Masanori and Yokomori, Fumiya and Ozawa, Kenji},
  journal={IEICE Transactions on Information and Systems},
  volume={99},
  number={7},
  pages={1877--1884},
  year={2016}
}
```

#### PSOLA
```
@article{moulines1990pitch,
  title={Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones},
  author={Moulines, Eric and Charpentier, Francis},
  journal={Speech communication},
  volume={9},
  number={5-6},
  pages={453--467},
  year={1990}
}
```

#### GPT-SoVITS
```
@misc{gptsovits2024,
  title={GPT-SoVITS: Few-shot Voice Conversion with GPT and SoVITS},
  author={RVC-Boss},
  year={2024},
  howpublished={\url{https://github.com/RVC-Boss/GPT-SoVITS}}
}
```

#### RVC
```
@misc{rvc2023,
  title={Retrieval-based Voice Conversion},
  author={RVC-Project},
  year={2023},
  howpublished={\url{https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI}}
}
```

#### kNN-VC
```
@inproceedings{vanniekerk2023knnvc,
  title={kNN-VC: Any-to-Any Voice Conversion with Self-Supervised Discrete Speech Representations},
  author={van Niekerk, Benjamin and Baas, Marc-Andr{\'e} and Kamper, Herman},
  booktitle={Interspeech},
  year={2023}
}
```

---

## Ethical Use Guidelines

### Voice Cloning Ethics

All voice conversion technologies mentioned in this project should be used responsibly:

1. **Consent**: Always obtain explicit consent before cloning someone's voice
2. **Transparency**: Disclose when voice content is synthetic or converted
3. **No Impersonation**: Do not use for fraud, impersonation, or deception
4. **Respect Privacy**: Do not use without permission for public figures
5. **Legal Compliance**: Follow local laws regarding voice synthesis and deepfakes

### Prohibited Uses

The following uses are explicitly prohibited:

- ❌ Creating deepfake content without consent
- ❌ Impersonating others for fraud or deception
- ❌ Generating synthetic evidence for legal proceedings
- ❌ Harassment or defamation using synthetic voices
- ❌ Bypassing voice authentication systems
- ❌ Creating non-consensual adult content
- ❌ Political manipulation or misinformation campaigns

### Recommended Use Cases

Ethical and permitted uses include:

- ✅ Personal voice assistants and accessibility tools
- ✅ Content creation with proper disclosure
- ✅ Academic research and education
- ✅ Voice restoration for medical patients
- ✅ Entertainment and artistic projects (with consent)
- ✅ Language learning and pronunciation training
- ✅ Video game character voices (fictional)

---

## License Compatibility Matrix

| This Project | Compatible With | Notes |
|-------------|-----------------|-------|
| MIT | MIT, BSD, Apache 2.0, AGPL | ✅ Permissive, widely compatible |
| MIT | GPL/LGPL | ✅ Compatible (can be combined) |
| MIT | Proprietary | ✅ Can be used in closed-source |

**Note**: When combining code from multiple projects, ensure all licenses are compatible. The most restrictive license typically applies to the combined work.

### Special Considerations

**AGPL-3.0 (SoftVC VITS)**:
- If you use SoftVC VITS code, your derivative work must also be AGPL-3.0
- Network use counts as distribution (must provide source code)
- Commercial use is allowed but with source disclosure requirements

**Recommendation**: If building a commercial product, consider using MIT-licensed alternatives (RVC, Seed-VC, kNN-VC) instead of AGPL-licensed SoftVC VITS.

---

## How to Comply with Licenses

### For Academic/Research Use

1. Cite all relevant papers (see [Research Papers](#academic-work))
2. Include copyright notices in code
3. Acknowledge contributors
4. Share modifications if required by license (AGPL)

### For Commercial Use

1. Review each dependency's license carefully
2. Include license files in distributions
3. Provide attribution in documentation
4. For AGPL code: Either use alternatives or disclose source
5. Obtain legal review for complex licensing situations

### For Redistribution

1. Include original LICENSE files
2. Preserve copyright notices
3. Document any modifications
4. List all third-party components
5. Ensure license compatibility

---

## Contact and Attribution

**This Project**:
- Author: Murugan R
- License: MIT License
- Repository: https://github.com/MuruganR96/VoiceConversion_Survey

**For Questions About Licensing**:
- Open an issue on GitHub
- Contact project maintainer
- Consult original project licenses (links above)

---

## Disclaimer

This document provides a summary of third-party licenses for informational purposes. It does not constitute legal advice. For authoritative licensing information, always refer to the original LICENSE files in each respective repository.

The project maintainers make no warranties regarding license compliance and are not responsible for how you use these tools. Users are solely responsible for ensuring compliance with all applicable licenses and laws.

---

**Last Updated**: January 24, 2026

**Document Version**: 1.0

---

## Quick Reference: License Types

### MIT License
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include license and copyright notice
- ✅ No copyleft (can use in proprietary software)

### BSD-3-Clause License
- ✅ Similar to MIT
- ✅ Permissive
- ⚠️ Cannot use contributor names for endorsement

### Apache 2.0 License
- ✅ Patent grant included
- ✅ Commercial use allowed
- ⚠️ Must state changes
- ⚠️ Must include NOTICE file if present

### AGPL-3.0 License
- ✅ Commercial use allowed
- ⚠️ Strong copyleft (derivatives must be AGPL)
- ⚠️ Network use = distribution (must provide source)
- ⚠️ Must disclose source code

---

**For detailed legal requirements, consult the full license texts in the respective repositories.**
