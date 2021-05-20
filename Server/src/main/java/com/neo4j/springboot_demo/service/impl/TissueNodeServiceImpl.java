package com.neo4j.springboot_demo.service.impl;

import com.neo4j.springboot_demo.entity.nodes.TissueNode;
import com.neo4j.springboot_demo.repository.TissueNodeRepository;
import com.neo4j.springboot_demo.service.TissueNodeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
@Service
public class TissueNodeServiceImpl implements TissueNodeService {

    @Autowired
    private TissueNodeRepository tissueNodeRepository;

    @Override
    public void saveTissueNodeByTissueName(String tissueName) {
        tissueNodeRepository.saveTissueNode(tissueName);
    }

    @Override
    public void deleteById(Long id) {
        tissueNodeRepository.deleteById(id);
    }

    @Override
    public List<TissueNode> listTissueNodes() {
        List<TissueNode> tissueNodeList= tissueNodeRepository.listTissueNodes();
        return tissueNodeList;
    }

    @Override
    public Optional<TissueNode> getTissueNodeById(Long id) {
        Optional<TissueNode> tissueNodeOptional = tissueNodeRepository.findById(id);
        return tissueNodeOptional;
    }

    @Override
    public Optional<TissueNode> getTissueNodeByTissueName(String tissueName) {
        Optional<TissueNode> tissueNodeOptional = Optional.ofNullable(tissueNodeRepository.getTissueNodeByTissueName(tissueName));
        return tissueNodeOptional;
    }

    @Override
    public List<TissueNode> getRelationWithTissueNode(String tissueName) {
        return tissueNodeRepository.getRelationWithTissueNode(tissueName);
    }

    @Override
    public void saveRelationWithTissueNode(String firstTissueName, String secondTissueName) {
        tissueNodeRepository.saveRelationWithTissueNode(firstTissueName, secondTissueName);
    }

    @Override
    public List<TissueNode> getRelationWithDiseaseNode(String diseaseName) {
        return tissueNodeRepository.getRelationWithDiseaseNode(diseaseName);
    }

}
