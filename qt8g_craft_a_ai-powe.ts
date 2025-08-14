// qt8g_craft_a_ai-powe.ts

import express, { Request, Response } from 'express';
import * as bodyParser from 'body-parser';
import * as cors from 'cors';
import { tensorflow, Tensor } from '@tensorflow/tfjs';

interface IDataPoint {
  input: number[];
  output: number[];
}

interface IModel {
  train: (data: IDataPoint[]) => void;
  predict: (input: number[]) => number[];
}

class AIModel implements IModel {
  private model: tensorflow.Sequential;

  constructor() {
    this.model = tensorflow.sequential();
    this.model.add(tensorflow.layers.dense({ units: 1, inputShape: [1] }));
    this.model.compile({ optimizer: tensorflow.optimizers.adam(), loss: 'meanSquaredError' });
  }

  train(data: IDataPoint[]) {
    const inputs: Tensor = tensorflow.tensor2d(data.map((dp) => dp.input));
    const outputs: Tensor = tensorflow.tensor2d(data.map((dp) => dp.output));
    this.model.fit(inputs, outputs, { epochs: 100 });
  }

  predict(input: number[]): number[] {
    const tensor: Tensor = tensorflow.tensor2d([input]);
    const output: Tensor = this.model.predict(tensor);
    return output.dataSync() as number[];
  }
}

const app = express();
app.use(bodyParser.json());
app.use(cors());

const model: IModel = new AIModel();

app.post('/train', (req: Request, res: Response) => {
  const data: IDataPoint[] = req.body;
  model.train(data);
  res.send({ message: 'Model trained successfully' });
});

app.post('/predict', (req: Request, res: Response) => {
  const input: number[] = req.body;
  const output: number[] = model.predict(input);
  res.send({ output });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});