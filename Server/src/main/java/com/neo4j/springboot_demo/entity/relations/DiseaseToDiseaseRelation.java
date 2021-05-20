package com.neo4j.springboot_demo.entity.relations;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.TargetNode;

import java.util.List;

public class DiseaseToDiseaseRelation {

    @Id
    @GeneratedValue
    private Long id;

    @TargetNode
    private DiseaseNode diseaseNode;

    private List<String> PMIDs;

    public DiseaseToDiseaseRelation(DiseaseNode diseaseNode, List<String> PMIDs) {
        this.diseaseNode = diseaseNode;
        this.PMIDs = PMIDs;
    }

    public void setPMIDs(List<String> PMIDs) {
        this.PMIDs = PMIDs;
    }
    public List<String> getPMIDs() {
        return PMIDs;
    }
}
