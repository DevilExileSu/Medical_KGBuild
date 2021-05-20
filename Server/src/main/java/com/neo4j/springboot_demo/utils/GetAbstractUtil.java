package com.neo4j.springboot_demo.utils;
import com.neo4j.springboot_demo.response.Result;
import com.neo4j.springboot_demo.response.ResultCode;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


import java.io.IOException;
import java.util.*;

public class GetAbstractUtil {
    public static String userAgent="Mozilla";
    public static Result getAbstract(String PMIDS) {
        Document doc;
        StringBuilder sb = new StringBuilder("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=");
        sb.append(PMIDS);
        sb.append("&retmode=xml");
        String url = sb.toString();
        List<Map> res = new ArrayList<>();
        try {
            doc = Jsoup.connect(url).userAgent(userAgent).get();
            Elements abstracts = doc.getElementsByTag( "Abstract" );
            Elements PMIDElement = doc.getElementsByTag("PMID");
            for(int i=0;i<abstracts.size(); i++) {
                Map<String, Object> tmp = new HashMap<>();
                List<String> text = new ArrayList<>();
                Elements abstractText = abstracts.tagName("AbstractText");
                for(int j=0;j<abstractText.size();j++) {
                    text.addAll(
                            Arrays.asList(abstractText
                                    .get(i)
                                    .text()
                                    .replaceAll("([a-z0-9])\\. ([A-Z])", "$1\\.\n$2")
                                    .split("\n")));
                }
                tmp.put("PMID",  PMIDElement.get(i).text());
                tmp.put("sents", text);
                res.add(tmp);
            }
            return new Result(ResultCode.SUCCESS, res);
        } catch (Exception e) {
            // TODO: handle exception
            System.out.println("getPMID  失败,url:"+url + " " + e.getMessage());
            return new Result(200003, "getPMID  失败,url:"+url + " " + e.getMessage(), false);
        }
    }
}
