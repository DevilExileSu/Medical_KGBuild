package com.neo4j.springboot_demo.response;

import java.io.Serializable;

public enum ResultCode {

    SUCCESS(true, 10000, "操作成功"),
    FAIL(false, 20000, "操作失败"),
    DATA_NOT_EXIST(false, 20001, "数据不存在"),
    SERVER_ERROR(false, 99999, "抱歉，系统繁忙，请稍后重试！");

    boolean success;
    int code;
    String message;
    ResultCode(boolean success, int code, String message) {
        this.success = success;
        this.code = code;
        this.message = message;
    }

    public boolean success() {
        return success;
    }

    public int code() {
        return code;
    }

    public String message() {
        return message;
    }
}
