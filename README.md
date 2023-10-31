# Read Me

<br/>

**Paper:** Identifying Conspiracy Theories News based on Event Relation Graph<br/>
**Accepted:** The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://arxiv.org/pdf/2310.18545.pdf

<br/>

## Event Relation Graph
* **Dataset:** We used MAVEN-ERE dataset for building the event relation graph (https://github.com/THU-KEG/MAVEN-ERE)<br/>
* **Data Splitting:** The document id of train/dev/test sets under media source splitting are in the ./LOCO/LOCO_media_split folder. The document id of train/dev/test sets under random splitting are in the ./LOCO/LOCO_random_split folder (section 4.2)<br/>
* **mavenere_event_relation_label.py:** the code for processing the event relations labels<br/>
* **event_relation_graph.py:** the code for training the event relation graph

<br/>

## Conspiracy Theories Identification
* **Dataset:** We used LOCO dataset for the conspiracy theories news identification experiments. (https://osf.io/snpcg/)<br/>
* **conspiracy_event_relation_graph.py:** the code for conspiracy news identification based on event relation graph

<br/>

## Citation

If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2023. Identifying Conspiracy Theories News based on Event Relation Graph. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore. Association for Computational Linguistics.

@misc{lei2023identifying,
      title={Identifying Conspiracy Theories News based on Event Relation Graph}, 
      author={Yuanyuan Lei and Ruihong Huang},
      year={2023},
      eprint={2310.18545},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

<br/>



