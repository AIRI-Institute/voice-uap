# Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy

This repository accompanies the paper **“Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy”** by Elvir Karimov, Alexander Varlamov, Danil Ivanov, Dmitrii Korzh, and Oleg Y. Rogov. The paper presents a new method for generating Universal Adversarial Patches (UAPs) in the audio domain to protect speaker identity, introducing a novel Exponential Total Variance loss function and a length‐independent UAP insertion procedure.

---

## Abstract

Deep learning voice models are commonly used nowadays, but the safety of processing of personal data, such as human identity and speech content, remains suspicious. To prevent malicious user identification, speaker obfuscation methods were proposed. Current methods, particularly based on universal adversarial patch (UAP) applications, have drawbacks such as significant degradation of audio quality, decreased speech recognition quality, low transferability across different voice biometrics models, and performance dependence on the input audio length. To mitigate these drawbacks, in this work, we introduce and leverage the novel Exponential Total Variance (TV) loss function and provide experimental evidence that it positively affects UAP strength and imperceptibility. Moreover, we present a novel scalable UAP insertion procedure and demonstrate its uniformly high performance for various audio lengths.

---

## Contributions

1. **Incorporation of the Novel Loss Function**  
   We propose a novel Exponential Total Variance (TV) loss function inspired by TV loss from the image domain, designed to preserve the imperceptibility of UAPs.

2. **Length-Independent UAP**  
   We introduce a length-independent UAP generation approach by training on long audio samples with a repeat padding strategy, making it effective for real-world applications.  
   _To the best of our knowledge, this strategy, although being well-known, has not been used in prior UAP training._

3. **Length-Agnostic Evaluation Procedure**  
   We establish a rigorous evaluation protocol that accounts for dataset biases, including variations in loudness levels. Furthermore, a proper padding strategy based on audio repetition is implemented to prevent the UAP from exploiting artificially silent segments, ensuring robustness across different audio lengths.

---

## Paper

- **Title:** Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy  
- **Authors:** Elvir Karimov, Alexander Varlamov, Danil Ivanov, Dmitrii Korzh, Oleg Y. Rogov  
- **arXiv:** [arXiv:2505.19951](https://arxiv.org/abs/2505.19951)  

---

## Citation

```bibtex
@misc{karimov2025novellossenhanceduniversaladversarial,
  title={Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy},
  author={Elvir Karimov and Alexander Varlamov and Danil Ivanov and Dmitrii Korzh and Oleg Y. Rogov},
  year={2025},
  eprint={2505.19951},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2505.19951},
}
