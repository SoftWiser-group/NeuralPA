[BugLab_Argument_Swapping]^return  ( RunnerImpl )  constructorParam.newInstance ( constructor ) ;^31^^^^^28^32^return  ( RunnerImpl )  constructor.newInstance ( constructorParam ) ;^[CLASS] RunnerProvider  [METHOD] newInstance [RETURN_TYPE] RunnerImpl   String runnerClassName Class<?> constructorParam [VARIABLES] Class  constructorParam  runnerClass  boolean  hasJUnit45OrHigher  String  runnerClassName  Constructor  constructor  
[BugLab_Variable_Misuse]^return  ( RunnerImpl )  3.newInstance ( constructorParam ) ;^31^^^^^28^32^return  ( RunnerImpl )  constructor.newInstance ( constructorParam ) ;^[CLASS] RunnerProvider  [METHOD] newInstance [RETURN_TYPE] RunnerImpl   String runnerClassName Class<?> constructorParam [VARIABLES] Class  constructorParam  runnerClass  boolean  hasJUnit45OrHigher  String  runnerClassName  Constructor  constructor  