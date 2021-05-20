package com.neo4j.springboot_demo.service.impl;

import com.neo4j.springboot_demo.entity.nodes.GeneNode;
import com.neo4j.springboot_demo.repository.GeneNodeRepository;
import com.neo4j.springboot_demo.service.GeneNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
@Service
public class GeneNodeServiceImpl implements GeneNodeService {

    @Autowired
    private GeneNodeRepository geneNodeRepository;

    @Override
    public void saveGeneNodeByGeneName(String geneName) {
        geneNodeRepository.saveGeneNode(geneName);
    }

    @Override
    public void deleteById(Long id) {
        geneNodeRepository.deleteById(id);
    }

    @Override
    public List<GeneNode> listGeneNodes() {
        return geneNodeRepository.listGeneNodes();
    }

    @Override
    public Optional<GeneNode> getGeneNodeById(Long id) {
        return geneNodeRepository.findById(id);
    }

    @Override
    public Optional<GeneNode> getGeneNodeByGeneName(String geneName) {
        GeneNode geneNode = geneNodeRepository.getGeneNodeByGeneName(geneName);
        return Optional.ofNullable(geneNode);
    }

    @Override
    public List<GeneNode> getRelationWithDiseaseNode(String diseaseName) {
        return geneNodeRepository.getRelationWithDiseaseNode(diseaseName);
    }
}
