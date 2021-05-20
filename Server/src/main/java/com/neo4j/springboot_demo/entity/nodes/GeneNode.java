package com.neo4j.springboot_demo.entity.nodes;

import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Node("GeneNode")
public class GeneNode implements Serializable {
//    private static final long serialVersionUID = 5495804314585571663L;
    @Id
    @GeneratedValue
    private Long id;
    private String geneName;

    public GeneNode() {};

    public Long getId() {
        return id;
    }

    public String getGeneName() {
        return geneName;
    }

    public void setGeneName(String geneName) {
        this.geneName = geneName;
    }
    @Relationship(type="gene_associated_with_disease", direction = Relationship.Direction.OUTGOING)
    private List<DiseaseNode> geneNodeList = new ArrayList<>();
}
