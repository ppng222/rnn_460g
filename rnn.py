import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_one_hot(sequence, vocab_size):
        encoding = np.zeros((1,len(sequence), vocab_size), dtype=np.float32)
        for i in range(len(sequence)):
            encoding[0, i, sequence[i]] = 1
        return encoding

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #Define the network!
		#Batch first defines where the batch parameter is in the tensor
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True).to(device)
        
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        
    def forward(self, x):
        hidden_state = self.init_hidden()
        #print(x.size())
        output,hidden_state = self.rnn(x, hidden_state)
        #print("RAW OUTPUT")
        #print(output.size())
		#Shouldn't need to resize if using batches, this eliminates the first dimension
        output = output.contiguous().view(-1, self.hidden_size)
        #print("REFORMED OUTPUT")
        #print(output.size())
        output = self.fc(output)
        #print("FC OUTPUT")
        #print(output.size())
        
        return output, hidden_state
        
    def init_hidden(self):
        #Hey,this is our hidden state. Hopefully if we don't have a batch it won't yell at us
        #Also a note, pytorch, by default, wants the batch index to be the middle dimension here. 
        #So it looks like (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return hidden

def main():
    
    data_path = "tiny-shakespeare.txt"
    with open(data_path) as handle:
        phrases = handle.readlines()
    chars = set(''.join(phrases))
    intChar = dict(enumerate(chars))
    charInt = {character: index for index, character in intChar.items()}

    input_sequence  = []
    target_sequence = []

    for i in range(len(phrases)):
        input_sequence.append(phrases[i][:-1])
        target_sequence.append(phrases[i][1:])

    for i in range(len(phrases)):
        input_sequence[i]  = [charInt[char] for char in input_sequence[i]]
        target_sequence[i] = [charInt[char] for char in target_sequence[i]]

    vocab_size = len(charInt)

    

    create_one_hot(input_sequence[0], vocab_size)

    model = RNNModel(vocab_size, vocab_size, 100, 1).to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for i in range(len(input_sequence)):
            if len(input_sequence[i]) == 0:
                continue    
            optimizer.zero_grad()
            x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size)).to(device)
            y = torch.Tensor(target_sequence[i]).to(device)
            output, hidden = model(x)

            lossValue = loss(output, y.view(-1).long())
            lossValue.backward()
            optimizer.step()

        print("Loss: {:.4f}".format(lossValue.item()))
        print(f"Epoch: {epoch}")
    def predict(model, character):
        characterInput = np.array([charInt[c] for c in character])
        #print(characterInput)
        characterInput = create_one_hot(characterInput, vocab_size)
        #print(characterInput)
        characterInput = torch.from_numpy(characterInput).to(device)
        #print(character)
        out, hidden = model(characterInput)
        
        #Get output probabilities
        
        prob = nn.functional.softmax(out[-1], dim=0).data
        #print(prob)
        character_index = torch.max(prob, dim=0)[1].item()
        
        return intChar[character_index], hidden
    def sample(model, out_len, start=' '):
        characters = [ch for ch in start]
        currentSize = out_len - len(characters)
        for i in range(currentSize):
            character, hidden_state = predict(model, characters)
            characters.append(character)
            
        return ''.join(characters)
    print(sample(model,50))
    return 0

if __name__ == "__main__":
    main()