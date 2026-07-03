# References

This page lists the literature basis reflected conceptually in the repository documentation and code comments. Reference PDFs are not committed to the repository.

## 1. Zheng et al. 2024

| Item | Description |
|---|---|
| Title | High accurate detection method for aortic valve opening of seismocardiography signals |
| Journal | Biomedical Signal Processing and Control, 2024 |
| Key contribution | SCG AO peak estimation using preprocessing, SVMD, waveform factor, and seventh-power detector |
| Why relevant | Motivates AO-focused SCG preprocessing and high-order AO enhancement concepts |
| Repository use | Zheng-style MTI, AO reconstruction, and seventh-power detector concepts are documented and partially adapted |
| Copying note | No figures or long text are copied; the repository uses conceptual adaptation |

## 2. Di Rienzo et al. 2017

| Item | Description |
|---|---|
| Title | An algorithm for the beat-to-beat assessment of cardiac mechanics during sleep on Earth and in microgravity from the seismocardiogram |
| Journal | Scientific Reports, 2017 |
| Key contribution | Beat-to-beat SCG cardiac mechanics analysis with artifact removal, SCG fiducial point extraction, congruency check, and CTI estimation |
| Why relevant | Supports SCG fiducial extraction and CTI-style beat-wise mechanics analysis |
| Repository use | SCG AO/AC reference generation and fiducial interpretation |
| Copying note | Concepts are summarized; no copyrighted figures are reproduced |

## 3. Qiao et al. 2022

| Item | Description |
|---|---|
| Title | Contactless multiscale measurement of cardiac motion using biomedical radar sensor |
| Journal | Frontiers in Cardiovascular Medicine, 2022 |
| Key contribution | Radar cardiac motion measurement model, micro-motion based RCG analysis, and ECG-RCG mechanical motion interpretation |
| Why relevant | Supports radar micro-motion interpretation and respiration/cardiac component separation |
| Repository use | Radar displacement, respiration suppression, and cardiac-band morphology documentation |
| Copying note | Radar equations and concepts are restated; original figures are not copied |

## 4. Ryu et al. 2026

| Item | Description |
|---|---|
| Title | Analysis of Aortic Valve Opening and Closure Using Cardiac Signals Acquired by Non-Contact FMCW Radar |
| Affiliation | Dankook University |
| Key contribution | ECG/SCG/FMCW radar simultaneous acquisition and beat-wise Radar AO/AC candidate timing comparison against SCG reference |
| Why relevant | This repository is the public research-code/documentation package for that analysis direction |
| Repository use | End-to-end pipeline organization, firmware documentation, and paper-ready export |
| Copying note | The repository documents methods and code; paper PDFs are not committed by default |
