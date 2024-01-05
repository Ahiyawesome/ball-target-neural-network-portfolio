using JetBrains.Annotations;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;

public class playerMan : MonoBehaviour
{
    public bool stop = false;
    private GameObject target;
    private GameObject spawnPos;

    private static int inputNum = 21;
    private static int hiddenTwoNum = 9;
    private static int outputNum = 8;

    public int[] inputs = new int[inputNum];
    private float[] hiddenLayerTwo = new float[hiddenTwoNum];
    public float[] outputs = new float[outputNum];
    public float[] probabilities = new float[outputNum];
    public int[] bestOutputs = new int[outputNum];

    private float[] firstWeights = new float[inputNum*hiddenTwoNum];           //(21*9)
    private float[] thirdWeights = new float[hiddenTwoNum*outputNum];           //(9*8)
    public float[] dfirstWeights = new float[inputNum * hiddenTwoNum];          //(21*9)
    public float[] dthirdWeights = new float[hiddenTwoNum * outputNum];          //(9*8)

    private float[] thirdErrors = new float[outputNum];

    private float lr = 0.01f;
    private int prevAction=-1;
    private float speed = 1;
    private float stepCount = 0;
    private float minStepCount = 1e9f;

    void Start()
    {
        for (int i=0; i<72; i++)
        {
            firstWeights[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
            thirdWeights[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
        }
        for (int i=64; i<189; i++)
        {
            firstWeights[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
        }
        target = GameObject.Find("Target");
        spawnPos = GameObject.Find("SpawnPos");
    }

    // Update is called once per frame
    void Update()
    {
        
        BestPossibleMove();
        ForwardPropagation(prevAction);           
        CalculateCost();
        MovePlayer();
        BackPropagation();
        ResetCheck();
        lr -= (lr > 0.001f ? 0.001f : 0);
        stepCount++;
        

    }
    void BestPossibleMove()
    {   
        for (int i=0; i<outputNum; i++)
        {
            bestOutputs[i] = 0;
        }
        if (target.transform.position.x < gameObject.transform.position.x) {
            if (target.transform.position.y <= gameObject.transform.position.y + 0.15 && target.transform.position.y >= gameObject.transform.position.y - 0.15)
            {
                bestOutputs[1] = 1;             // left
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                bestOutputs[6] = 1;             // left-up
            }
            else if (target.transform.position.y < gameObject.transform.position.y)
            {
                bestOutputs[7] = 1;             // left-down
            }
        }
        else if (target.transform.position.x > gameObject.transform.position.x)
        {
            if (target.transform.position.y < gameObject.transform.position.y)
            {
                bestOutputs[5] = 1;             // right-down
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                bestOutputs[4] = 1;             // right-up
            }
            else if (target.transform.position.y == gameObject.transform.position.y)
            {
                bestOutputs[0] = 1;             // right
            }
        }
        else
        {
            if (target.transform.position.y < gameObject.transform.position.y)
            {
                bestOutputs[3] = 1;             // down
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                bestOutputs[2] = 1;             // up                      
            }
        }
    }
    void ForwardPropagation(int prevAct)
    {

        int weightIndex = 0;


        // Initialize inputs

        for (int i = 0; i < 8; i++)
        {
            inputs[i] = 0;
        }
        if (target.transform.position.x < gameObject.transform.position.x)
        {
            if (target.transform.position.y <= gameObject.transform.position.y + 0.1 && target.transform.position.y >= gameObject.transform.position.y - 0.1)
            {
                inputs[1] = 1;             // left
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                inputs[6] = 1;             // left-up
            }
            else if (target.transform.position.y < gameObject.transform.position.y)
            {
                inputs[7] = 1;             // left-down
            }
        }
        else if (target.transform.position.x > gameObject.transform.position.x)
        {
            if (target.transform.position.y < gameObject.transform.position.y)
            {
                inputs[5] = 1;             // right-down
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                inputs[4] = 1;             // right-up
            }
            else if (target.transform.position.y == gameObject.transform.position.y)
            {
                inputs[0] = 1;             // right
            }
        }
        else
        {
            if (target.transform.position.y < gameObject.transform.position.y)
            {
                inputs[3] = 1;             // down
            }
            else if (target.transform.position.y > gameObject.transform.position.y)
            {
                inputs[2] = 1;             // up                      
            }
        }

        for (int i = 8; i < 16; i++)
        {
            if (prevAct == i) inputs[i] = 1;
            else inputs[i] = 0;

        }

        if (gameObject.transform.position.y > 2.3) inputs[16] = 1;       // close to top bound
        else inputs[16] = 0;
        if (gameObject.transform.position.y < -2.6) inputs[17] = 1;     // close to bottom bound
        else inputs[17] = 0;
        if (gameObject.transform.position.x > 6) inputs[18] = 1;      // close to right bound
        else inputs[18] = 0;
        if (gameObject.transform.position.x < -6.2) inputs[19] = 1;     // close to left bound
        else inputs[19] = 0;

        inputs[20] = 1;                                                 // bias layer

        // Calculate nodes into second hidden layer
        weightIndex = 0;
        for (int i = 0; i < hiddenTwoNum; i++)
        {
            float node = 0;
            float activate;
            for (int j = 0; j < inputNum; j++)
            {
                node += inputs[j] * firstWeights[weightIndex];
                weightIndex++;
            }
            activate = sigmoid(node);
            hiddenLayerTwo[i] = activate;
        }

        hiddenLayerTwo[8] = 1;                                          // bias layer

        // Calculate Outputs
        weightIndex = 0;
        for (int i=0; i<outputNum; i++) {
            float node = 0;
            float activate;
            for (int j=0; j<hiddenTwoNum; j++)
            {
                node += hiddenLayerTwo[j] * thirdWeights[weightIndex];
                weightIndex++;
            }
            activate = sigmoid(node);
            outputs[i] = activate;
        }

        // Use softmax for probabilites
        float sum = 0;

        for (int i=0; i<outputNum; i++)
        {
            sum += Mathf.Exp(outputs[i]);
        }
        if (sum == 0) sum = 1f;          // get rid of divide by zero error
        for (int i=0; i< outputNum; i++)
        {
            float temp;
            temp = Mathf.Exp(outputs[i]) / sum;
            probabilities[i] = temp;
        }
    }

    void CalculateCost()
    {

        //Output Layer Costs
        for (int i = 0; i < outputNum; i++)
        {
            thirdErrors[i] = probabilities[i] - bestOutputs[i];
        }
    }

    void MovePlayer()
    {
        float max = -1e9f, tempi=-1;
        for (int i=0; i< outputNum; i++)
        {
            if (outputs[i] > max)
            {
                max = outputs[i];
                tempi = i;
            }
        }
        if (tempi == 0) // move right
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(speed * Time.deltaTime, 0, 0);
            prevAction = 8;
            Debug.Log("right");
        }
        if (tempi == 1) // move left
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(-speed * Time.deltaTime, 0, 0);
            prevAction = 9;
            Debug.Log("left");
        }
        if (tempi == 2) // move up
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(0, speed * Time.deltaTime, 0);
            prevAction = 10;
            Debug.Log("up");
        }
        if (tempi == 3) // move down
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(0, -speed * Time.deltaTime, 0);
            prevAction = 11;
            Debug.Log("down");
        }
        if (tempi == 4) // move right-up
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(speed * Time.deltaTime, speed * Time.deltaTime, 0);
            prevAction = 12;
            Debug.Log("right-up");
        }
        if (tempi == 5) // move right-down
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(speed * Time.deltaTime, -speed * Time.deltaTime, 0);
            prevAction = 13;
            Debug.Log("right-down");
        }
        if (tempi == 6) // move left-up
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(-speed * Time.deltaTime, speed * Time.deltaTime, 0);
            prevAction = 14;
            Debug.Log("left-up");
        }
        if (tempi == 7) // move left-down
        {
            gameObject.transform.position = gameObject.transform.position + new Vector3(-speed * Time.deltaTime, -speed * Time.deltaTime, 0);
            prevAction = 15;
            Debug.Log("left-down");
        }
    }

    void ResetCheck()
    {
        if (gameObject.transform.position.y > 2.78f || gameObject.transform.position.y < -3.03 || gameObject.transform.position.x > 6.31 || gameObject.transform.position.x < -6.56 || gameObject.transform.position == target.transform.position)
        {
            Reset();
        }
    }

    void Reset()
    {
        gameObject.transform.position = spawnPos.transform.position;
        stepCount = 0;
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (stepCount < minStepCount)
        {
            minStepCount = stepCount;
        }
        gameObject.transform.position = spawnPos.transform.position;
        stepCount = 0;
    }

    void BackPropagation ()  /* Calculates error by taking the derivative of each weight with respect to the probabilities, and updates the weight very slightly 
                                according to that gradient */
    {
        // Calculating gradient or smthn
        int weightIndex = 0;
        for (int i=0; i< outputNum; i++)
        {
            for (int j=0; j<hiddenTwoNum; j++)
            {
                dthirdWeights[weightIndex] = (probabilities[i] - bestOutputs[i]) * (probabilities[i] * (1 - probabilities[i])) * hiddenLayerTwo[j];
                weightIndex++;
            }
        }

        weightIndex = 0;
        for (int i = 0; i < hiddenTwoNum; i++)
        {
            for (int j = 0; j < inputNum; j++)
            {
                for (int k=0; k<outputNum; k++)
                {
                    dfirstWeights[weightIndex] += (probabilities[k] - bestOutputs[k]) * (probabilities[k] * (1 - probabilities[k])) * thirdWeights[i + (k * 8)] * (hiddenLayerTwo[i] * (1 - hiddenLayerTwo[i])) * inputs[j];
                }

                weightIndex++;
            }
        }

        // Updating Weights and Biases based on Gradient?
        for (int i=0; i<72; i++)
        {
            firstWeights[i] -= lr * dfirstWeights[i];
            thirdWeights[i] -= lr * dthirdWeights[i];
        }
        for (int i = 72; i < 189; i++)
        {
            firstWeights[i] -= lr * dfirstWeights[i];
        }

    }

    float sigmoid(float input)
    {
        float t = (MathF.Exp(input) + 1);
        if (t == 0)
        {
            t = 1f;              // get rid of divide by zero error
        }

        return (MathF.Exp(input) / t);
    }
}
