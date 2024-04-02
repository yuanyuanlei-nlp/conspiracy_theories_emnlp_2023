# Read Me

<br/>

**Paper:** Identifying Conspiracy Theories News based on Event Relation Graph<br/>
**Accepted:** The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://aclanthology.org/2023.findings-emnlp.656/

<br/>

## Event Relation Graph
* **Dataset:** We used MAVEN-ERE dataset for building the event relation graph (https://github.com/THU-KEG/MAVEN-ERE)<br/>
* **mavenere_event_relation_label.py:** the code for processing the event relations labels<br/>
* **event_relation_graph.py:** the code for training the event relation graph

<br/>

## Conspiracy Theories Identification
* **Dataset:** We used LOCO dataset for the conspiracy theories news identification experiments. (https://osf.io/snpcg/)<br/>
* **Data Splitting:** The document id of train/dev/test sets under media source splitting are in the ./LOCO/LOCO_media_split folder. The document id of train/dev/test sets under random splitting are in the ./LOCO/LOCO_random_split folder (section 4.2)<br/>
* **conspiracy_event_relation_graph.py:** the code for conspiracy news identification based on event relation graph

<br/>

## Citation

If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2023. Identifying Conspiracy Theories News based on Event Relation Graph. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9811–9822, Singapore. Association for Computational Linguistics.

```bibtex
@inproceedings{lei-huang-2023-identifying,
    title = "Identifying Conspiracy Theories News based on Event Relation Graph",
    author = "Lei, Yuanyuan  and
      Huang, Ruihong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.656",
    doi = "10.18653/v1/2023.findings-emnlp.656",
    pages = "9811--9822",
    abstract = "Conspiracy theories, as a type of misinformation, are narratives that explains an event or situation in an irrational or malicious manner. While most previous work examined conspiracy theory in social media short texts, limited attention was put on such misinformation in long news documents. In this paper, we aim to identify whether a news article contains conspiracy theories. We observe that a conspiracy story can be made up by mixing uncorrelated events together, or by presenting an unusual distribution of relations between events. Achieving a contextualized understanding of events in a story is essential for detecting conspiracy theories. Thus, we propose to incorporate an event relation graph for each article, in which events are nodes, and four common types of event relations, coreference, temporal, causal, and subevent relations, are considered as edges. Then, we integrate the event relation graph into conspiracy theory identification in two ways: an event-aware language model is developed to augment the basic language model with the knowledge of events and event relations via soft labels; further, a heterogeneous graph attention network is designed to derive a graph embedding based on hard labels. Experiments on a large benchmark dataset show that our approach based on event relation graph improves both precision and recall of conspiracy theory identification, and generalizes well for new unseen media sources.",
}
```

<br/>




