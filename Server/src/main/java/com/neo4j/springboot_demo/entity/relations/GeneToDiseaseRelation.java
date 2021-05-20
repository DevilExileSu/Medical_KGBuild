package com.neo4j.springboot_demo.entity.relations;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import org.springframework.data.neo4j.core.schema.*;


@RelationshipProperties
public class GeneToDiseaseRelation {
    @Id
    @GeneratedValue
    private Long id;

    @TargetNode
    private DiseaseNode diseaseNode;

    public GeneToDiseaseRelation(DiseaseNode diseaseNode) {
        this.diseaseNode = diseaseNode;
    }
}
