[BugLab_Variable_Misuse]^_classMappings.put ( new ClassKey ( this ) , inst ) ;^37^^^^^34^39^_classMappings.put ( new ClassKey ( forType ) , inst ) ;^[CLASS] SimpleValueInstantiators  [METHOD] addValueInstantiator [RETURN_TYPE] SimpleValueInstantiators   Class<?> forType ValueInstantiator inst [VARIABLES] Class  forType  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  inst  
[BugLab_Argument_Swapping]^_classMappings.put ( new ClassKey ( inst ) , forType ) ;^37^^^^^34^39^_classMappings.put ( new ClassKey ( forType ) , inst ) ;^[CLASS] SimpleValueInstantiators  [METHOD] addValueInstantiator [RETURN_TYPE] SimpleValueInstantiators   Class<?> forType ValueInstantiator inst [VARIABLES] Class  forType  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  inst  
[BugLab_Argument_Swapping]^ValueInstantiator inst = beanDesc.get ( new ClassKey ( _classMappings.getBeanClass (  )  )  ) ;^45^^^^^42^47^ValueInstantiator inst = _classMappings.get ( new ClassKey ( beanDesc.getBeanClass (  )  )  ) ;^[CLASS] SimpleValueInstantiators  [METHOD] findValueInstantiator [RETURN_TYPE] ValueInstantiator   DeserializationConfig config BeanDescription beanDesc ValueInstantiator defaultInstantiator [VARIABLES] DeserializationConfig  config  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  defaultInstantiator  inst  BeanDescription  beanDesc  
[BugLab_Variable_Misuse]^ValueInstantiator inst = null.get ( new ClassKey ( beanDesc.getBeanClass (  )  )  ) ;^45^^^^^42^47^ValueInstantiator inst = _classMappings.get ( new ClassKey ( beanDesc.getBeanClass (  )  )  ) ;^[CLASS] SimpleValueInstantiators  [METHOD] findValueInstantiator [RETURN_TYPE] ValueInstantiator   DeserializationConfig config BeanDescription beanDesc ValueInstantiator defaultInstantiator [VARIABLES] DeserializationConfig  config  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  defaultInstantiator  inst  BeanDescription  beanDesc  
[BugLab_Argument_Swapping]^return  ( defaultInstantiator == null )  ? inst : inst;^46^^^^^42^47^return  ( inst == null )  ? defaultInstantiator : inst;^[CLASS] SimpleValueInstantiators  [METHOD] findValueInstantiator [RETURN_TYPE] ValueInstantiator   DeserializationConfig config BeanDescription beanDesc ValueInstantiator defaultInstantiator [VARIABLES] DeserializationConfig  config  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  defaultInstantiator  inst  BeanDescription  beanDesc  
[BugLab_Wrong_Operator]^return  ( inst != null )  ? defaultInstantiator : inst;^46^^^^^42^47^return  ( inst == null )  ? defaultInstantiator : inst;^[CLASS] SimpleValueInstantiators  [METHOD] findValueInstantiator [RETURN_TYPE] ValueInstantiator   DeserializationConfig config BeanDescription beanDesc ValueInstantiator defaultInstantiator [VARIABLES] DeserializationConfig  config  boolean  HashMap  _classMappings  long  serialVersionUID  ValueInstantiator  defaultInstantiator  inst  BeanDescription  beanDesc  