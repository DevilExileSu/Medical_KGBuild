package com.neo4j.springboot_demo.entity.nodes;


import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Node("DiseaseNode")
public class DiseaseNode implements Serializable {
//    private static final long serialVersionUID = 5495806402785571663L;
    @Id
    @GeneratedValue
    private Long id;
    private String diseaseName;

    public DiseaseNode() {};

    public Long getId() {
        return id;
    }

    public String getDiseaseName() {
        return diseaseName;
    }

    public void setDiseaseName(String diseaseName) {
        this.diseaseName = diseaseName;
    }


    @Relationship(type="gene_associated_with_disease", direction = Relationship.Direction.INCOMING)
    private List<GeneNode> geneNodeList = new ArrayList<>();

    @Relationship(type="disease_associated_with_disease", direction = Relationship.Direction.INCOMING)
    private List<DiseaseNode> diseaseNodeList = new ArrayList<>();

    @Relationship(type="disease_associated_with_tissue", direction = Relationship.Direction.OUTGOING)
    private List<TissueNode> tissueNodeList = new ArrayList<>();
}
