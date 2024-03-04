[P14_Delete_Statement]^^71^^^^^70^72^return ExceptionClosure.<E>exceptionClosure (  ) ;^[CLASS] ClosureUtils  [METHOD] exceptionClosure [RETURN_TYPE] <E>   [VARIABLES] boolean  
[P14_Delete_Statement]^^84^^^^^83^85^return NOPClosure.<E>nopClosure (  ) ;^[CLASS] ClosureUtils  [METHOD] nopClosure [RETURN_TYPE] <E>   [VARIABLES] boolean  
[P5_Replace_Variable]^return TransformerClosure.transformerClosure ( this ) ;^99^^^^^98^100^return TransformerClosure.transformerClosure ( transformer ) ;^[CLASS] ClosureUtils  [METHOD] asClosure [RETURN_TYPE] <E>   Transformer<? super E, ?> transformer [VARIABLES] Transformer  transformer  boolean  
[P14_Delete_Statement]^^99^100^^^^98^100^return TransformerClosure.transformerClosure ( transformer ) ; }^[CLASS] ClosureUtils  [METHOD] asClosure [RETURN_TYPE] <E>   Transformer<? super E, ?> transformer [VARIABLES] Transformer  transformer  boolean  
[P5_Replace_Variable]^return ForClosure.forClosure (  closure ) ;^115^^^^^114^116^return ForClosure.forClosure ( count, closure ) ;^[CLASS] ClosureUtils  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  int  count  
[P5_Replace_Variable]^return ForClosure.forClosure ( count ) ;^115^^^^^114^116^return ForClosure.forClosure ( count, closure ) ;^[CLASS] ClosureUtils  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  int  count  
[P5_Replace_Variable]^return ForClosure.forClosure ( closure, count ) ;^115^^^^^114^116^return ForClosure.forClosure ( count, closure ) ;^[CLASS] ClosureUtils  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  int  count  
[P14_Delete_Statement]^^115^116^^^^114^116^return ForClosure.forClosure ( count, closure ) ; }^[CLASS] ClosureUtils  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  int  count  
[P3_Replace_Literal]^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^131^^^^^130^132^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^[CLASS] ClosureUtils  [METHOD] whileClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> closure [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure (  closure, false ) ;^131^^^^^130^132^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^[CLASS] ClosureUtils  [METHOD] whileClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> closure [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure ( predicate,  false ) ;^131^^^^^130^132^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^[CLASS] ClosureUtils  [METHOD] whileClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> closure [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure ( closure, predicate, false ) ;^131^^^^^130^132^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^[CLASS] ClosureUtils  [METHOD] whileClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> closure [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P14_Delete_Statement]^^131^^^^^130^132^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^[CLASS] ClosureUtils  [METHOD] whileClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> closure [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P3_Replace_Literal]^return WhileClosure.<E>whileClosure ( predicate, closure, false ) ;^148^^^^^146^149^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^[CLASS] ClosureUtils  [METHOD] doWhileClosure [RETURN_TYPE] <E>   Closure<? super E> closure Predicate<? super E> predicate [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure (  closure, true ) ;^148^^^^^146^149^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^[CLASS] ClosureUtils  [METHOD] doWhileClosure [RETURN_TYPE] <E>   Closure<? super E> closure Predicate<? super E> predicate [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure ( predicate,  true ) ;^148^^^^^146^149^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^[CLASS] ClosureUtils  [METHOD] doWhileClosure [RETURN_TYPE] <E>   Closure<? super E> closure Predicate<? super E> predicate [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P5_Replace_Variable]^return WhileClosure.<E>whileClosure ( closure, predicate, true ) ;^148^^^^^146^149^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^[CLASS] ClosureUtils  [METHOD] doWhileClosure [RETURN_TYPE] <E>   Closure<? super E> closure Predicate<? super E> predicate [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P14_Delete_Statement]^^148^^^^^146^149^return WhileClosure.<E>whileClosure ( predicate, closure, true ) ;^[CLASS] ClosureUtils  [METHOD] doWhileClosure [RETURN_TYPE] <E>   Closure<? super E> closure Predicate<? super E> predicate [VARIABLES] boolean  Closure  closure  Predicate  predicate  
[P7_Replace_Invocation]^return chainedClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName )  ) ;^165^^^^^163^166^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName [VARIABLES] boolean  String  methodName  
[P14_Delete_Statement]^^165^166^^^^163^166^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName )  ) ; }^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName [VARIABLES] boolean  String  methodName  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer (  paramTypes, args )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName,  args )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( args, paramTypes, methodName )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, args, paramTypes )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P7_Replace_Invocation]^return invokerClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P5_Replace_Variable]^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( paramTypes, methodName, args )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P8_Replace_Mix]^return invokerClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, 3, args )  ) ;^186^^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ;^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P14_Delete_Statement]^^186^187^^^^183^187^return asClosure ( InvokerTransformer.<E, Object>invokerTransformer ( methodName, paramTypes, args )  ) ; }^[CLASS] ClosureUtils  [METHOD] invokerClosure [RETURN_TYPE] <E>   String methodName Class<?>[] paramTypes Object[] args [VARIABLES] Class[]  paramTypes  boolean  String  methodName  Object[]  args  
[P7_Replace_Invocation]^return ChainedClosure .chainedClosure (  )  ;^202^^^^^201^203^return ChainedClosure.chainedClosure ( closures ) ;^[CLASS] ClosureUtils  [METHOD] chainedClosure [RETURN_TYPE] <E>    closures [VARIABLES] Closure[]  closures  boolean  
[P14_Delete_Statement]^^202^^^^^201^203^return ChainedClosure.chainedClosure ( closures ) ;^[CLASS] ClosureUtils  [METHOD] chainedClosure [RETURN_TYPE] <E>    closures [VARIABLES] Closure[]  closures  boolean  
[P5_Replace_Variable]^return ChainedClosure.chainedClosure ( null ) ;^220^^^^^219^221^return ChainedClosure.chainedClosure ( closures ) ;^[CLASS] ClosureUtils  [METHOD] chainedClosure [RETURN_TYPE] <E>   Closure<? super E>> closures [VARIABLES] boolean  Collection  closures  
[P8_Replace_Mix]^return ChainedClosure .chainedClosure (  )  ;^220^^^^^219^221^return ChainedClosure.chainedClosure ( closures ) ;^[CLASS] ClosureUtils  [METHOD] chainedClosure [RETURN_TYPE] <E>   Closure<? super E>> closures [VARIABLES] boolean  Collection  closures  
[P14_Delete_Statement]^^220^^^^^219^221^return ChainedClosure.chainedClosure ( closures ) ;^[CLASS] ClosureUtils  [METHOD] chainedClosure [RETURN_TYPE] <E>   Closure<? super E>> closures [VARIABLES] boolean  Collection  closures  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure (  trueClosure ) ;^239^^^^^237^240^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure [VARIABLES] boolean  Closure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( predicate ) ;^239^^^^^237^240^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure [VARIABLES] boolean  Closure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( trueClosure, predicate ) ;^239^^^^^237^240^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure [VARIABLES] boolean  Closure  trueClosure  Predicate  predicate  
[P8_Replace_Mix]^return IfClosure.<E>ifClosure ( predicate, this ) ;^239^^^^^237^240^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure [VARIABLES] boolean  Closure  trueClosure  Predicate  predicate  
[P14_Delete_Statement]^^239^240^^^^237^240^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ; }^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure [VARIABLES] boolean  Closure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure (  trueClosure, falseClosure ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( predicate,  falseClosure ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( predicate, trueClosure ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( trueClosure, predicate, falseClosure ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( predicate, falseClosure, trueClosure ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return IfClosure.<E>ifClosure ( falseClosure, trueClosure, predicate ) ;^259^^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ;^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P14_Delete_Statement]^^259^260^^^^256^260^return IfClosure.<E>ifClosure ( predicate, trueClosure, falseClosure ) ; }^[CLASS] ClosureUtils  [METHOD] ifClosure [RETURN_TYPE] <E>   Predicate<? super E> predicate Closure<? super E> trueClosure Closure<? super E> falseClosure [VARIABLES] boolean  Closure  falseClosure  trueClosure  Predicate  predicate  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure (  closures, null ) ;^282^^^^^280^283^return SwitchClosure.<E>switchClosure ( predicates, closures, null ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures [VARIABLES] Closure[]  closures  boolean  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( predicates,  null ) ;^282^^^^^280^283^return SwitchClosure.<E>switchClosure ( predicates, closures, null ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures [VARIABLES] Closure[]  closures  boolean  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( closures, predicates, null ) ;^282^^^^^280^283^return SwitchClosure.<E>switchClosure ( predicates, closures, null ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures [VARIABLES] Closure[]  closures  boolean  Predicate[]  predicates  
[P8_Replace_Mix]^return SwitchClosure.<E>switchClosure ( predicates, closures, false ) ;^282^^^^^280^283^return SwitchClosure.<E>switchClosure ( predicates, closures, null ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures [VARIABLES] Closure[]  closures  boolean  Predicate[]  predicates  
[P14_Delete_Statement]^^282^^^^^280^283^return SwitchClosure.<E>switchClosure ( predicates, closures, null ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures [VARIABLES] Closure[]  closures  boolean  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure (  closures, defaultClosure ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( predicates,  defaultClosure ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( predicates, closures ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( closures, predicates, defaultClosure ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P5_Replace_Variable]^return SwitchClosure.<E>switchClosure ( predicates, defaultClosure, closures ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P8_Replace_Mix]^return SwitchClosure.<E>switchClosure ( predicates, null, defaultClosure ) ;^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P14_Delete_Statement]^^308^^^^^305^309^return SwitchClosure.<E>switchClosure ( predicates, closures, defaultClosure ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Predicate<? super E>[] predicates Closure<? super E>[] closures Closure<? super E> defaultClosure [VARIABLES] Closure[]  closures  boolean  Closure  defaultClosure  Predicate[]  predicates  
[P14_Delete_Statement]^^333^^^^^332^334^return SwitchClosure.switchClosure ( predicatesAndClosures ) ;^[CLASS] ClosureUtils  [METHOD] switchClosure [RETURN_TYPE] <E>   Closure<E>> predicatesAndClosures [VARIABLES] boolean  Map  predicatesAndClosures  
[P2_Replace_Operator]^if  ( objectsAndClosures != null )  {^356^^^^^355^370^if  ( objectsAndClosures == null )  {^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^if  ( 4 == null )  {^356^^^^^355^370^if  ( objectsAndClosures == null )  {^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P8_Replace_Mix]^if  ( objectsAndClosures == false )  {^356^^^^^355^370^if  ( objectsAndClosures == null )  {^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("The object and closure map must not be null");^356^357^358^^^355^370^if  ( objectsAndClosures == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P16_Remove_Block]^^356^357^358^^^355^370^if  ( objectsAndClosures == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P13_Insert_Block]^if  ( objectsAndClosures == null )  {     throw new IllegalArgumentException ( "The object and closure map must not be null" ) ; }^357^^^^^355^370^[Delete]^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P8_Replace_Mix]^final Closure<? super E> def = objectsAndClosures .size (  )  ;^359^^^^^355^370^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final Closure<? super E>[] trs = new Closure[size];final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^359^^^^^355^370^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final int size = objectsAndClosures.size (  ) ;final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^359^^^^^355^370^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^359^^^^^355^370^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P1_Replace_Type]^final  long  size = objectsAndClosures.size (  ) ;^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P3_Replace_Literal]^final int size = objectsAndClosures.size() + 0 ;^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^final int size = objectsAndClosures.remove (  ) ;^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;final int size = objectsAndClosures.size (  ) ;^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P12_Insert_Condition]^if  ( objectsAndClosures == null )  { final int size = objectsAndClosures.size (  ) ; }^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P3_Replace_Literal]^final int size = objectsAndClosures.size() + 3 ;^360^^^^^355^370^final int size = objectsAndClosures.size (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^360^361^^^^355^370^final int size = objectsAndClosures.size (  ) ; final Closure<? super E>[] trs = new Closure[size];^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final Closure<? super E> def = objectsAndClosures.remove ( null ) ;final Closure<? super E>[] trs = new Closure[size];^361^^^^^355^370^final Closure<? super E>[] trs = new Closure[size];^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final Predicate<E>[] preds = new Predicate[size];final Closure<? super E>[] trs = new Closure[size];^361^^^^^355^370^final Closure<? super E>[] trs = new Closure[size];^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P11_Insert_Donor_Statement]^final Closure<? super E>[] trs = new Closure[size];final Predicate<E>[] preds = new Predicate[size];^362^^^^^355^370^final Predicate<E>[] preds = new Predicate[size];^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P1_Replace_Type]^long  i = 0;^363^^^^^355^370^int i = 0;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P3_Replace_Literal]^int i = i;^363^^^^^355^370^int i = 0;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^preds[i] = EqualPredicate.<E>equalPredicate ( entry .getValue (  )   ) ;^365^^^^^355^370^preds[i] = EqualPredicate.<E>equalPredicate ( entry.getKey (  )  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P8_Replace_Mix]^preds[i] ;^365^^^^^355^370^preds[i] = EqualPredicate.<E>equalPredicate ( entry.getKey (  )  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^trs[i] = this.getValue (  ) ;^366^^^^^355^370^trs[i] = entry.getValue (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^trs[i] = entry .getKey (  )  ;^366^^^^^355^370^trs[i] = entry.getValue (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P8_Replace_Mix]^trs[i]  =  trs[i] ;^366^^^^^355^370^trs[i] = entry.getValue (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^for  ( final Map.Entry<? extends E, Closure<E>> entry : objectsAndClosures.remove (  )  )  {^364^^^^^355^370^for  ( final Map.Entry<? extends E, Closure<E>> entry : objectsAndClosures.entrySet (  )  )  {^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^for  ( final Map.Entry<? extends E, Closure<E>> entry : objectsAndClosures .remove ( 4 )   )  {^364^^^^^355^370^for  ( final Map.Entry<? extends E, Closure<E>> entry : objectsAndClosures.entrySet (  )  )  {^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^364^365^366^367^368^355^370^for  ( final Map.Entry<? extends E, Closure<E>> entry : objectsAndClosures.entrySet (  )  )  { preds[i] = EqualPredicate.<E>equalPredicate ( entry.getKey (  )  ) ; trs[i] = entry.getValue (  ) ; i++; }^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^365^^^^^355^370^preds[i] = EqualPredicate.<E>equalPredicate ( entry.getKey (  )  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^366^^^^^355^370^trs[i] = entry.getValue (  ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure (  trs, def ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure ( preds,  def ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure ( preds, trs ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure ( def, trs, preds ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure ( trs, preds, def ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P7_Replace_Invocation]^return ClosureUtils.<E>ifClosure ( preds, trs, def ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P5_Replace_Variable]^return ClosureUtils.<E>switchClosure ( preds, def, trs ) ;^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
[P14_Delete_Statement]^^369^^^^^355^370^return ClosureUtils.<E>switchClosure ( preds, trs, def ) ;^[CLASS] ClosureUtils  [METHOD] switchMapClosure [RETURN_TYPE] <E>   Closure<E>> objectsAndClosures [VARIABLES] Closure[]  trs  Entry  entry  boolean  Closure  def  Predicate[]  preds  Map  objectsAndClosures  int  i  size  
