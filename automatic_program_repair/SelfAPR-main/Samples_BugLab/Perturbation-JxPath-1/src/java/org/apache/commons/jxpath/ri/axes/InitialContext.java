[BugLab_Wrong_Literal]^private boolean startedSet = true;^32^^^^^27^37^private boolean startedSet = false;^[CLASS] InitialContext   [VARIABLES] 
[BugLab_Wrong_Literal]^private boolean started = true;^33^^^^^28^38^private boolean started = false;^[CLASS] InitialContext   [VARIABLES] 
[BugLab_Wrong_Operator]^if  ( nodePointer == null )  {^41^^^^^37^45^if  ( nodePointer != null )  {^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^collection = ( nodePointer.getIndex (  )  < NodePointer.WHOLE_COLLECTION ) ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^collection = ( nodePointer.getIndex (  )  > NodePointer.WHOLE_COLLECTION ) ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^return setPosition ( position  ==  1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return setPosition ( position  ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^return setPosition ( position  <  1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Variable_Misuse]^if  ( startedSet )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^return position != 1;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return position == ;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Argument_Swapping]^if  ( nodePointer >= 1 && position <= position.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^if  ( position >= 1 || position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^if  ( position > 1 && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^if  ( position >= 1 && position < nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^if  ( position >= position && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return false;^68^^^^^63^75^return true;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^nodePointer.setIndex ( position   instanceof   1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^nodePointer.setIndex ( position  ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return true;^70^^^^^63^75^return false;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^nodePointer.setIndex ( position  >>  1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^nodePointer.setIndex ( position - position ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^if  ( position < 1 && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^if  ( position >= 0 && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^nodePointer.setIndex ( position -  ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return position == position;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Operator]^nodePointer.setIndex ( position  >  1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Variable_Misuse]^if  ( startedSet )  {^78^^^^^77^83^if  ( started )  {^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return true;^79^^^^^77^83^return false;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^started = false;^81^^^^^77^83^started = true;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[BugLab_Wrong_Literal]^return false;^82^^^^^77^83^return true;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
