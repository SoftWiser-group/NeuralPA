[BugLab_Variable_Misuse]^this.left = right;^38^^^^^37^39^this.left = left;^[CLASS] Add  [METHOD] setLeftOperand [RETURN_TYPE] void   Evaluation left [VARIABLES] Evaluation  left  right  boolean  
[BugLab_Variable_Misuse]^this.right = left;^42^^^^^41^43^this.right = right;^[CLASS] Add  [METHOD] setRightOperand [RETURN_TYPE] void   Evaluation right [VARIABLES] Evaluation  left  right  boolean  
[BugLab_Argument_Swapping]^return left.evaluate ( context.doubleValue ( context )  + context.doubleValue ( right ) ) ;^46^47^48^^^45^49^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Argument_Swapping]^return context.evaluate ( context.doubleValue ( right )  + context.doubleValue ( left ) ) ;^46^47^48^^^45^49^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Wrong_Operator]^return context.evaluate ( context.doubleValue ( left )   ==  context.doubleValue ( right ) ) ;^46^47^48^^^45^49^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Variable_Misuse]^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( left ) ) ;^46^47^48^^^45^49^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Wrong_Operator]^return context.evaluate ( context.doubleValue ( left )   &&  context.doubleValue ( right ) ) ;^46^47^48^^^45^49^return context.evaluate ( context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Variable_Misuse]^context.doubleValue ( right )  + context.doubleValue ( right ) ) ;^47^48^^^^45^49^context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Argument_Swapping]^context.doubleValue ( context )  + left.doubleValue ( right ) ) ;^47^48^^^^45^49^context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Variable_Misuse]^context.doubleValue ( left )  + context.doubleValue ( left ) ) ;^47^48^^^^45^49^context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
[BugLab_Argument_Swapping]^context.doubleValue ( left )  + right.doubleValue ( context ) ) ;^47^48^^^^45^49^context.doubleValue ( left )  + context.doubleValue ( right ) ) ;^[CLASS] Add  [METHOD] evaluate [RETURN_TYPE] Evaluation   EvaluationContext context [VARIABLES] Evaluation  left  right  EvaluationContext  context  boolean  
