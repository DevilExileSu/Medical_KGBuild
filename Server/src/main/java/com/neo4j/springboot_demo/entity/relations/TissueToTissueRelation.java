package com.neo4j.springboot_demo.entity.relations;

import com.neo4j.springboot_demo.entity.nodes.DiseaseNode;
import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.TargetNode;

public class TissueToTissueRelation {
    @Id
    @GeneratedValue
    private Long id;

    @TargetNode
    private TissueNode tissueNode;


    public TissueToTissueRelation(TissueNode tissueNode) {
        this.tissueNode = tissueNode;
    }
}
