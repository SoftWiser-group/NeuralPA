[BugLab_Wrong_Literal]^private boolean startedSetIteration = true;^50^^^^^45^55^private boolean startedSetIteration = false;^[CLASS] EvalContext 1   [VARIABLES] 
[BugLab_Wrong_Literal]^private boolean done = true;^51^^^^^46^56^private boolean done = false;^[CLASS] EvalContext 1   [VARIABLES] 
[BugLab_Wrong_Literal]^private boolean hasPerformedIteratorStep = true;^52^^^^^47^57^private boolean hasPerformedIteratorStep = false;^[CLASS] EvalContext 1   [VARIABLES] 
[BugLab_Variable_Misuse]^return  (  ( Comparable )  o2 ) .compareTo ( o2 ) ;^59^^^^^58^60^return  (  ( Comparable )  o2 ) .compareTo ( o1 ) ;^[CLASS] EvalContext 1  [METHOD] compare [RETURN_TYPE] int   Object o1 Object o2 [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Object  o1  o2  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( parentContext != null || parentContext.isChildOrderingRequired (  )  )  {^86^^^^^85^90^if  ( parentContext != null && parentContext.isChildOrderingRequired (  )  )  {^[CLASS] EvalContext 1  [METHOD] getDocumentOrder [RETURN_TYPE] int   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( parentContext == null && parentContext.isChildOrderingRequired (  )  )  {^86^^^^^85^90^if  ( parentContext != null && parentContext.isChildOrderingRequired (  )  )  {^[CLASS] EvalContext 1  [METHOD] getDocumentOrder [RETURN_TYPE] int   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return ;^87^^^^^85^90^return 1;^[CLASS] EvalContext 1  [METHOD] getDocumentOrder [RETURN_TYPE] int   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return position;^89^^^^^85^90^return 0;^[CLASS] EvalContext 1  [METHOD] getDocumentOrder [RETURN_TYPE] int   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( getDocumentOrder (  )  == 0 )  {^100^^^^^97^104^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] isChildOrderingRequired [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( getDocumentOrder (  )  != -1 )  {^100^^^^^97^104^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] isChildOrderingRequired [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return false;^101^^^^^97^104^return true;^[CLASS] EvalContext 1  [METHOD] isChildOrderingRequired [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return true;^103^^^^^97^104^return false;^[CLASS] EvalContext 1  [METHOD] isChildOrderingRequired [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( pointerIterator == null )  {^110^^^^^109^123^if  ( pointerIterator != null )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( getDocumentOrder (  )  == 0 )  {^114^^^^^109^123^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( getDocumentOrder (  )  != 1 )  {^114^^^^^109^123^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( !startedSetIteration && !hasPerformedIteratorStep )  {^118^^^^^109^123^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( !done && !startedSetIteration )  {^118^^^^^109^123^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( !done || !hasPerformedIteratorStep )  {^118^^^^^109^123^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( pointerIterator == null )  {^129^^^^^128^149^if  ( pointerIterator != null )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( getDocumentOrder (  )  == 0 )  {^133^^^^^128^149^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( getDocumentOrder (  )  != position )  {^133^^^^^128^149^if  ( getDocumentOrder (  )  != 0 )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( !startedSetIteration && !hasPerformedIteratorStep )  {^140^^^^^128^149^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( !done && !startedSetIteration )  {^140^^^^^128^149^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( !done || !hasPerformedIteratorStep )  {^140^^^^^128^149^if  ( !done && !hasPerformedIteratorStep )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( startedSetIteration )  {^143^^^^^128^149^if  ( done )  {^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^hasPerformedIteratorStep = true;^146^^^^^128^149^hasPerformedIteratorStep = false;^[CLASS] EvalContext 1  [METHOD] next [RETURN_TYPE] Object   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^done = false;^155^^^^^154^168^done = true;^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^done = true;^162^^^^^154^168^done = false;^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( position != 0 || nextNode (  )  )  {^156^^^^^154^168^if  ( position != 0 && nextNode (  )  )  {^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( position == 0 && nextNode (  )  )  {^156^^^^^154^168^if  ( position != 0 && nextNode (  )  )  {^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( position != position && nextNode (  )  )  {^156^^^^^154^168^if  ( position != 0 && nextNode (  )  )  {^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^done = true;^157^^^^^154^168^done = false;^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^hasPerformedIteratorStep = false;^167^^^^^154^168^hasPerformedIteratorStep = true;^[CLASS] EvalContext 1  [METHOD] performIteratorStep [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return true;^192^^^^^178^203^return false;^[CLASS] EvalContext 1  [METHOD] constructIterator [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  ArrayList  list  NodePointer  pointer  int  position  Comparator  REVERSE_COMPARATOR  HashSet  set  
[BugLab_Wrong_Operator]^if  ( getDocumentOrder (  )  != 1 )  {^195^^^^^178^203^if  ( getDocumentOrder (  )  == 1 )  {^[CLASS] EvalContext 1  [METHOD] constructIterator [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  ArrayList  list  NodePointer  pointer  int  position  Comparator  REVERSE_COMPARATOR  HashSet  set  
[BugLab_Wrong_Literal]^if  ( getDocumentOrder (  )  == position )  {^195^^^^^178^203^if  ( getDocumentOrder (  )  == 1 )  {^[CLASS] EvalContext 1  [METHOD] constructIterator [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  ArrayList  list  NodePointer  pointer  int  position  Comparator  REVERSE_COMPARATOR  HashSet  set  
[BugLab_Argument_Swapping]^Collections.sort ( REVERSE_COMPARATOR, list ) ;^199^^^^^178^203^Collections.sort ( list, REVERSE_COMPARATOR ) ;^[CLASS] EvalContext 1  [METHOD] constructIterator [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  ArrayList  list  NodePointer  pointer  int  position  Comparator  REVERSE_COMPARATOR  HashSet  set  
[BugLab_Wrong_Literal]^return false;^202^^^^^178^203^return true;^[CLASS] EvalContext 1  [METHOD] constructIterator [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  ArrayList  list  NodePointer  pointer  int  position  Comparator  REVERSE_COMPARATOR  HashSet  set  
[BugLab_Wrong_Operator]^if  ( pos == 0 )  {^211^^^^^209^225^if  ( pos != 0 )  {^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( pos != position )  {^211^^^^^209^225^if  ( pos != 0 )  {^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( position != 0 )  {^218^^^^^209^225^if  ( pos != 0 )  {^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( pos >= 0 )  {^218^^^^^209^225^if  ( pos != 0 )  {^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( pos != -1 )  {^218^^^^^209^225^if  ( pos != 0 )  {^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^setPosition ( position ) ;^219^^^^^209^225^setPosition ( pos ) ;^[CLASS] EvalContext 1  [METHOD] getContextNodeList [RETURN_TYPE] List   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  List  list  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( pos != 0 )  {^233^^^^^232^247^if  ( position != 0 )  {^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( position == 0 )  {^233^^^^^232^247^if  ( position != 0 )  {^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^if  ( position != -1 )  {^233^^^^^232^247^if  ( position != 0 )  {^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: "  &&  "should not request pointer list while "  &&  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while "  ||  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: "  >>  "should not request pointer list while "  >>  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while "   instanceof   "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: "  >  "should not request pointer list while "  >  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while "  &&  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: "  ^  "should not request pointer list while "  ^  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while "  ==  "iterating over an EvalContext" ) ;^234^235^236^237^^232^247^throw new JXPathException ( "Simultaneous operations: " + "should not request pointer list while " + "iterating over an EvalContext" ) ;^[CLASS] EvalContext 1  [METHOD] getNodeSet [RETURN_TYPE] NodeSet   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  BasicNodeSet  set  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( ptr != null )  {^260^^^^^258^266^if  ( ptr == null )  {^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context [" + getPosition (  &&  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context [" + getPosition (  >>  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context ["  |  getPosition (  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context [" + getPosition (  >  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context [" + getPosition (  <=  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^return "Expression context ["  &&  getPosition (  )  + "] " + ptr.asPath (  ) ;^264^^^^^258^266^return "Expression context [" + getPosition (  )  + "] " + ptr.asPath (  ) ;^[CLASS] EvalContext 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  Pointer  ptr  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Operator]^if  ( rootContext != null )  {^273^^^^^272^277^if  ( rootContext == null )  {^[CLASS] EvalContext 1  [METHOD] getRootContext [RETURN_TYPE] RootContext   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^position = ;^283^^^^^282^284^position = 0;^[CLASS] EvalContext 1  [METHOD] reset [RETURN_TYPE] void   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^return pos;^287^^^^^286^288^return position;^[CLASS] EvalContext 1  [METHOD] getCurrentPosition [RETURN_TYPE] int   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return false;^324^^^^^314^344^return true;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^if  ( !hasPerformedIteratorStep )  {^320^^^^^314^344^if  ( !startedSetIteration )  {^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^startedSetIteration = false;^321^^^^^314^344^startedSetIteration = true;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return true;^327^^^^^314^344^return false;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return false;^333^^^^^314^344^return true;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return false;^340^^^^^314^344^return true;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return true;^343^^^^^314^344^return false;^[CLASS] EvalContext 1  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^this.position = pos;^359^^^^^358^361^this.position = position;^[CLASS] EvalContext 1  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Wrong_Literal]^return false;^360^^^^^358^361^return true;^[CLASS] EvalContext 1  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] RootContext  rootContext  boolean  done  hasPerformedIteratorStep  startedSetIteration  EvalContext  parentContext  Iterator  pointerIterator  int  pos  position  Comparator  REVERSE_COMPARATOR  
[BugLab_Variable_Misuse]^return  (  ( Comparable )  o2 ) .compareTo ( o2 ) ;^59^^^^^58^60^return  (  ( Comparable )  o2 ) .compareTo ( o1 ) ;^[CLASS] 1  [METHOD] compare [RETURN_TYPE] int   Object o1 Object o2 [VARIABLES] boolean  Object  o1  o2  