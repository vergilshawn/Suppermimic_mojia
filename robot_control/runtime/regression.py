"""Evidence-based early stop: persistent deterministic physical regression, not WM loss."""
class RegressionGuard:
    def __init__(self,patience=3,return_drop=.15,completion_drop=.20):
        self.patience=patience;self.return_drop=return_drop;self.completion_drop=completion_drop
        self.best_return=float('-inf');self.best_completion=0.;self.bad_count=0
    def observe(self,result):
        if result.get('status')!='complete' or not result.get('evaluation_is_full'):raise ValueError('Need successful full-clip evaluation')
        score=result['normalized_return'];completion=result['completion_rate']
        improved=score>self.best_return
        if improved:self.best_return=score;self.best_completion=completion;self.bad_count=0
        elif self.best_return>.2 and (score<self.best_return-self.return_drop or completion<self.best_completion-self.completion_drop):self.bad_count+=1
        else:self.bad_count=0
        return improved,self.bad_count>=self.patience
