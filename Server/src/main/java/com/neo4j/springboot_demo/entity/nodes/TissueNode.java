package com.neo4j.springboot_demo.entity.nodes;

import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Node("TissueNode")
public class TissueNode implements Serializable {
//    private static final long serialVersionUID = 5495804314585571873L;
    @Id
    @GeneratedValue
    private Long id;

    private String tissueName;

    public TissueNode() {};

    public Long getId() {
        return id;
    }

    public String getTissueName() {
        return tissueName;
    }

    public void setTissueName(String tissueName) {
        this.tissueName = tissueName;
    }

    @Relationship(type="tissue_associated_with_tissue", direction = Relationship.Direction.INCOMING)
    private List<TissueNode> tissueNodeList = new ArrayList<>();
    @Relationship(type="disease_associated_with_tissue", direction = Relationship.Direction.INCOMING)
    private List<DiseaseNode>  diseaseNodeList= new ArrayList<>();
}
