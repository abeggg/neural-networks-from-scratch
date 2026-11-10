from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[1]
n=runpy.run_path(str(root/'backprop_ml.py'))
x,y=n['load_dataset'](root/'classification2.txt')
splits=n['stratified_split'](x,y,seed=42)
scaler=n['Standardizer']().fit(splits[0][0])
model=n['MLPBinaryClassifier'](2,(5,4),seed=1,learning_rate=.05,l2=0)
result=n['gradient_check'](model,scaler.transform(splits[0][0][:8]),splits[0][1][:8])
assert result['relative_error'] < 1e-7, result
print(result)
