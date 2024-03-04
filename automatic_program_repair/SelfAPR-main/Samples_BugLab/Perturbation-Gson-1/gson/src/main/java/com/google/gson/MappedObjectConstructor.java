[BugLab_Argument_Swapping]^InstanceCreator<T> creator =  ( InstanceCreator<T> )  typeOfT.getHandlerFor ( instanceCreatorMap ) ;^49^^^^^48^54^InstanceCreator<T> creator =  ( InstanceCreator<T> )  instanceCreatorMap.getHandlerFor ( typeOfT ) ;^[CLASS] MappedObjectConstructor  [METHOD] construct [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InstanceCreator  creator  
[BugLab_Wrong_Operator]^if  ( creator == null )  {^50^^^^^48^54^if  ( creator != null )  {^[CLASS] MappedObjectConstructor  [METHOD] construct [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InstanceCreator  creator  
[BugLab_Argument_Swapping]^return typeOfT.createInstance ( creator ) ;^51^^^^^48^54^return creator.createInstance ( typeOfT ) ;^[CLASS] MappedObjectConstructor  [METHOD] construct [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InstanceCreator  creator  
[BugLab_Argument_Swapping]^return Array.newInstance ( TypeUtils.toRawClass ( length ) , type ) ;^57^^^^^56^58^return Array.newInstance ( TypeUtils.toRawClass ( type ) , length ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructArray [RETURN_TYPE] Object   Type type int length [VARIABLES] Type  type  boolean  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  int  length  
[BugLab_Wrong_Operator]^if  ( constructor != null )  {^63^^^^^60^78^if  ( constructor == null )  {^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  &  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "   instanceof   typeOfT   instanceof   " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  |  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  <  typeOfT  <  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  !=  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ||  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  ^  typeOfT  ^  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  >  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  ==  typeOfT  ==  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  <<  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Variable_Misuse]^return 4.newInstance (  ) ;^67^^^^^60^78^return constructor.newInstance (  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  |  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  &&  typeOfT  &&  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  ==  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ^  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  ^  typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  <<  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  ||  typeOfT  ||  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Argument_Swapping]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + e + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , typeOfT ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  <  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  &  typeOfT  &  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  >=  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Argument_Swapping]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + e + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , typeOfT ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (   instanceof   ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  >>  typeOfT  >>  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  >  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Argument_Swapping]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + e + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , typeOfT ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  |  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  |  typeOfT  |  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  <  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  !=  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "No-args constructor for "  !=  typeOfT  !=  " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^64^65^^^^60^78^throw new RuntimeException (  ( "No-args constructor for " + typeOfT + " does not exist. " + "Register an InstanceCreator with Gson for this type to fix this problem." )  ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  |  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  |  typeOfT  |  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  >  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^69^70^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ==  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  <  typeOfT  <  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  &  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^72^73^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  <=  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  ^  typeOfT  ^  ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new RuntimeException (  ( "Unable to invoke no-args constructor for "  ^  typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^75^76^^^^60^78^throw new RuntimeException (  ( "Unable to invoke no-args constructor for " + typeOfT + ". " + "Register an InstanceCreator with Gson for this type may fix this problem." ) , e ) ;^[CLASS] MappedObjectConstructor  [METHOD] constructWithNoArgConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InvocationTargetException  e  InstantiationException  e  IllegalAccessException  e  
[BugLab_Wrong_Literal]^AccessibleObject.setAccessible ( declaredConstructors, false ) ;^85^^^^^81^92^AccessibleObject.setAccessible ( declaredConstructors, true ) ;^[CLASS] MappedObjectConstructor  [METHOD] getNoArgsConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Constructor[]  declaredConstructors  Class  clazz  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  TypeInfo  typeInfo  
[BugLab_Wrong_Operator]^if  ( constructor.getParameterTypes (  ) .length != 0 )  {^87^^^^^81^92^if  ( constructor.getParameterTypes (  ) .length == 0 )  {^[CLASS] MappedObjectConstructor  [METHOD] getNoArgsConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Constructor[]  declaredConstructors  Class  clazz  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  TypeInfo  typeInfo  
[BugLab_Wrong_Literal]^if  ( constructor.getParameterTypes (  ) .length == -1 )  {^87^^^^^81^92^if  ( constructor.getParameterTypes (  ) .length == 0 )  {^[CLASS] MappedObjectConstructor  [METHOD] getNoArgsConstructor [RETURN_TYPE] <T>   Type typeOfT [VARIABLES] Type  typeOfT  boolean  Constructor  constructor  Constructor[]  declaredConstructors  Class  clazz  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  TypeInfo  typeInfo  
[BugLab_Argument_Swapping]^if  ( typeOfT.hasSpecificHandlerFor ( instanceCreatorMap )  )  {^102^^^^^101^106^if  ( instanceCreatorMap.hasSpecificHandlerFor ( typeOfT )  )  {^[CLASS] MappedObjectConstructor  [METHOD] register [RETURN_TYPE] <T>   Type typeOfT InstanceCreator<? extends T> creator [VARIABLES] Type  typeOfT  boolean  Logger  log  ParameterizedTypeHandlerMap  instanceCreatorMap  instanceCreators  InstanceCreator  creator  
