# Neural Program Analysis

This repository aims to provide a framework for neural network-based program analysis, making it easy for people to build neural networks and experiment with them.

# Table of contents
- [Automatic Program Repair](#automatic-program-repair)
- [Type Inference](#type-inference)

# Automatic Program Repair
| Year | Venue        | Paper                                                        | Code                                                         |
| ---- | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2023 | ESEC/FSE 2023 | Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair |               |
| 2022 | EMNLP 2022   | Using Developer Discussions to Guide Fixing Bugs in Software |                                                              |
| 2022 | ASE 2022     | SelfAPR: Self-supervised Program Repair with Test Execution Diagnostics |                                                   |
| 2022 | ICSE 2022    | Neural Program Repair with Execution-based Backpropagation   |                                                              |
| 2021 | PLDI 2021    | Learning to Find Naming Issues with Big Code and Small Supervision |                                                        |
| 2021 | FSE 2021     | A Syntax-Guided Edit Decoder for Neural Program Repair       |                                                              |
| 2021 | NeurIPS 2021 | Self-Supervised Bug Detection and Repair                     |                                                              |
| 2021 | ICML 2021    | TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer |                                                       |
| 2020 | ICSE 2020    | DLFix: Context-based Code Transformation Learning for Automated Program Repair |                                            |
| 2020 | ICLR 2020    | Hoppity: Learning Bug Detection and Repair                   |                                                              |
| 2019 | TSE 2019     | SequenceR: Sequence-to-Sequence Learning for End-to-End Program Repair |                                                    |
| 2019 | ESEC/FSE 2019 | DeepDelta: learning to repair compilation errors            |                                                              |
| 2019 | ICLR 2019    | Neural Program Repair by Jointly Learning to Localize and Repair |                                                          |

## Datasets
| ID |     Name           |     Language    |    #Bugs    |  Test Suite  |  Training  |  Testing   |     Links      |       Others      |
| -- | ------------------ | --------------- | ----------- | ------------ | ---------- | ---------- | -------------- | ----------------- |
| 1  | Bears              | Java            | 251         | yes          | yes        | yes        | [Github](https://github.com/bears-bugs/bears-benchmark) |                   |
| 2  | BFP medium         | Java            | 65454       | no           | yes        | yes        | [Zenodo](https://zenodo.org/records/7478730) |                   |
| 3  | BFP small          | Java            | 58350       | no           | yes        | yes        | [Zenodo](https://zenodo.org/records/7478730) |                   |
| 4  | BigFix             | Java            | 1.824 M     | no           | yes        | yes        | [Github](https://github.com/OOPSLA-2019-BugDetection/OOPSLA-2019-BugDetection) |  |
| 5  | Bugs2Fix           | Java            | 92849       | no           | yes        | yes        |  |                   |
| 6  | Bugs.jar           | Java            | 1158        | yes          | yes        | yes        | [Github](https://github.com/bugs-dot-jar/bugs-dot-jar) |                   |
| 7  | Code-Change-Data   | Java            | 44372       | no           | yes        | yes        | [Google Drive](https://drive.google.com/file/d/1wSl_SN17tbATqlhNMO0O7sEkH9gqJ9Vr/edit) |                   |
| 8  | CodeXGlue          | Java            | 122 K       | no           | no         | yes        |                |                   |
| 9  | CodRep             | Java            | 58069       | no           | yes        | yes        | [Github](https://github.com/ASSERT-KTH/CodRep) |                   |
| 10 | CPatMiner          | Java            | 44 K        | no           | yes        | yes        |                |                   |
| 11 | DeepRepair         | Java            | 374         | no           | yes        | no         |                |                   |
| 12 | Defects4J          | Java            | 835         | yes          | yes        | yes        | [Github](https://github.com/rjust/defects4j) |                   |
| 13 | Function-SStuBs4J  | Java            | 21047       | no           | yes        | yes        | [Zenodo](https://zenodo.org/records/5353354) |                   |
| 14 | IntroClassJava     | Java            | 998         | yes          | yes        | yes        | [Github](https://github.com/Spirals-Team/IntroClassJava) |                   |
| 15 | Java-med           | Java            | 7454        | no           | yes        | no         | [AWS](https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz) |                   |
| 16 | ManySStuBs4J large | Java            | 63923       | no           | yes        | yes        | [Zenodo](https://zenodo.org/record/3653444) |                   |
| 17 | ManySStuBs4J small | Java            | 10231       | no           | yes        | yes        | [Zenodo](https://zenodo.org/record/3653444) |                   |
| 18 | MegaDiff           | Java            | 663029      | no           | yes        | no         | [Zenodo](https://zenodo.org/record/5013515) |                   |
| 19 | Ponta              | Java            | 624         | no           | yes        | yes        | [Github](https://github.com/SAP/project-kb/tree/main/MSR2019) |                   |
| 20 | Pull-Request-Data  | Java            | 10666       | no           | yes        | yes        | [Zenodo](https://zenodo.org/records/7482720) |                   |
| 21 | Ratchet            | Java            | 35 K        | no           | yes        | yes        | [Github](https://github.com/hideakihata/NMTbasedCorrectivePatchGenerationDataset) |                   |
| 22 | Recoder            | Java            | 103585      | no           | yes        | no         | [Google Drive](https://drive.google.com/drive/folders/1ECNX98qj9FMdRT2MXOUY6aQ6-sNT0b_a) |                   |
| 23 | TRANSFER           | Java            | 408091      | no           | yes        | no         | [MEGA](https://mega.nz/file/u0wQzRga#Q2BHCuRD2aW_61vshVbcxj-ObYh2cyGhqOAmAXNn-T0) |                   |
| 24 | Mesbah             | Java            | 4.8 M       | no           | yes        | yes        |                |                   |
| 25 | AOJ                | C               | 2482        | no           | yes        | yes        | [Others](http://developers.u-aizu.ac.jp/index) |                   |
| 26 | Big-Vul            | C               | 3745        | no           | yes        | yes        | [Github](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) |                   |
| 27 | Code4Bench         | C               | 25 K        | yes          | yes        | yes        | [Zenodo](https://zenodo.org/record/2582968)               |                   |
| 28 | CodeHunt           | C               | 195 K       | yes          | yes        | yes        |                |                   |
| 29 | CVEFixes           | C               | 8482        | no           | yes        | yes        | [Github](https://github.com/secureIT-project/CVEfixes) |                   |
| 30 | DeepFix            | C               | 6971        | yes          | yes        | yes        | [Github](https://github.com/C-Ritam98/DeepFix) |                   |
| 31 | ManyBugs           | C               | 185         | yes          | yes        | yes        | [Others](https://repairbenchmarks.cs.umass.edu/) |                   |
| 32 | Prophet            | C               | 69          | yes          | yes        | yes        | [Github](https://github.com/epicosy/prophet) |                   |
| 33 | Prutor             | C               | 6971        | yes          | yes        | yes        | [Others](https://www.cse.iitk.ac.in/users/karkare/prutor/) |                   |
| 34 | BugAID             | JS              | 105133      | no           | yes        | yes        | [Others](http://salt.ece.ubc.ca/software/bugaid/) |                   |
| 35 | BugsJS             | JS              | 453         | yes          | yes        | yes        | [Others](https://bugsjs.github.io/#nav-download) |                   |
| 36 | HOPPITY            | JS              | 363 K       | no           | yes        | yes        | [Github](https://github.com/AI-nstein/hoppity) |                   |
| 37 | KATANA             | JS              | 114 K       | no           | yes        | yes        | [Github](https://github.com/saltlab/Katana) |                   |
| 38 | REPTORY            | JS              | 407 K       | no           | yes        | yes        | [Github](https://github.com/annon-reptory/reptory)               |                   |
| 39 | TFix               | JS              | 100 K       | no           | yes        | yes        | [Github](https://github.com/eth-sri/TFixs) |                   |
| 40 | ETH Py150          | Python          | 150 K       | no           | yes        | yes        | [Others](https://www.sri.inf.ethz.ch/py150) |                   |
| 41 | GitHub-Python      | Python          | 3 M         | no           | yes        | yes        | [Github](https://github.com/michiyasunaga/bifi) |                   |
| 42 | Mester             | Python          | 13 K        | no           | yes        | yes        |  |                   |
| 43 | PyPIBug            | Python          | 2374        | no           | yes        | yes        | [Github](https://github.com/microsoft/neurips21-self-supervised-bug-detection-and-repair) |                   |
| 44 | SSB-9M             | Python          | 9 M         | no           | yes        | no         | [Zenodo](https://zenodo.org/records/5845439) |                   |
| 45 | VUDENC             | Python          | 10 K        | no           | yes        | yes        | [Zenodo](https://zenodo.org/record/3559203) |                   |
| 46 | Chhatbar           | Python          | 286         | yes          | no         | yes        | [Github](https://github.com/purushottamkar/macer) |                   |
| 47 | SPoC               | C++             | 1835        | yes          | yes        | yes        | [Others](https://sumith1896.github.io/spoc/) |                   |
| 48 | QuixBugs           | Java Python     | 40          | yes          | yes        | yes        | [Github](https://github.com/jkoppel/QuixBugs) |                   |
| 49 | DeepDebug          | Java Python     | 523         | no           | yes        | yes        |                |                   |
| 50 | MSR20              | C C++           | 188 K       | no           | yes        | yes        | [Zenodo](https://zenodo.org/records/6324846) |                   |
| 51 | CoCoNut            | C Java JS Python| 14 M        | yes          | yes        | no         | [Github](https://github.com/lin-tan/CoCoNut-Artifact) |                   |
| 52 | CodeFlaw           | C Python        | 3902        | yes          | yes        | yes        | [Others](https://codeflaws.github.io/) |                   |
| 53 | ENCORE             | Java Python JS C++| 9.2 M     | no           | yes        | no         |                |                   |



# Type Inference
| Year | Venue        | Paper                                                        | Data Format                     | Code                       |
| ---- | ------------ | ------------------------------------------------------------ | ------------------------------- | -------------------------- |
| 2020 | PLDI 2020    | Typilus: Neural Type Hints                                   | Graph                           |                            |
| 2020 | ICLR 2020    | LAMBDANET: Probabilistic Type Inference Using Graph Neural Networks | Graph                    |                            |
| 2018 | ICLR 2018    | Learning to Represent Programs with Graphs                   | Graph                           |                            |
| 2023 | ASE  2023    | Generative Type Inference for Python                         | Text                            |                            |
| 2023 | ICLR 2023    | TypeT5: Seq2seq Type Inference using Static Analysis         | Text                            |                            |
| 2022 | ICSE 2022    | Type4Py: Practical Deep Similarity Learning-Based Type Inference for Python | Text             |                            |
| 2022 | ICSE 2022    | Static Inference Meets Deep Learning: A Hybrid Type Inference Approach for Python | Text       |                            |
| 2022 | TSE  2022    | Learning To Predict User-Defined Types                       | Text                            |                            |
| 2021 | NeurIPS 2021 | Type Inference as Optimization                               | Text                            |                            |
| 2018 | ESEC/FSE 2018| Deep Learning Type Inference                                 | Text                            |                            |
