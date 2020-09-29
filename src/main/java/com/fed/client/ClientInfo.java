package com.fed.client;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedHashMap;

public class ClientInfo {

    private String deviceId;//用户id
    private WebSocket webSocket;
    private ClientHandshake clientHandshake;
    private double income;
    private LinkedHashMap<String, INDArray> updateWeight;

    //手机硬件
    private String cpuType;//CPU型号
    private double ram;//内存总容量,单位GB
    private double storage;//外存空余容量，单位GB
    //...其他设备信息

    public Reputation clientReputation=new Reputation();//用户声誉


    //设备硬件评价
    public double getDeviceScore(){
        return 0.8*cpuScore()+0.15*ramScore()+0.05*storageScore();
    }


    //用户总体评价
    public double getClientSocre(){
        return 0.5*getDeviceScore()+0.5*clientReputation.calculateReputation();
    }

    //用户奖励计算
    public double getReword(double modelScore){
        double reward=getClientSocre()*modelScore;
        income+=reward;
        return reward;
    }


    private double cpuScore(){

        return 1.0;
    }

    private double ramScore(){
        double limit=8.0;//ram容量满分为8GB
        if(ram>=limit)
            return 1.0;
        else
            return ram/limit;
    }

    private double storageScore(){
        double limit=5.0;//storage容量满分5GB
        if(ram>=limit)
            return 1.0;
        else
            return ram/limit;
    }


    //setter
    public void setClientHandshake(ClientHandshake clientHandshake) {
        this.clientHandshake = clientHandshake;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public void setWebSocket(WebSocket webSocket) {
        this.webSocket = webSocket;
    }

    public void setUpdateWeight(LinkedHashMap<String, INDArray> updateWeight) {
        this.updateWeight = updateWeight;
    }

    public void setCpuType(String cpuType) {
        this.cpuType = cpuType;
    }

    public void setRam(double ram) {
        this.ram = ram;
    }

    public void setStorage(double storage) {
        this.storage = storage;
    }

    //getter
    public ClientHandshake getClientHandshake() {
        return clientHandshake;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public WebSocket getWebSocket() {
        return webSocket;
    }

    public LinkedHashMap<String, INDArray> getUpdateWeight() {
        return updateWeight;
    }

    public double getRam() {
        return ram;
    }

    public double getStorage() {
        return storage;
    }

    public double getIncome() {
        return income;
    }

    public String getCpuType() {
        return cpuType;
    }
}
