[BugLab_Wrong_Literal]^private int position = 1;^40^^^^^35^45^private int position = 0;^[CLASS] JDOMAttributeIterator   [VARIABLES] 
[BugLab_Variable_Misuse]^if  ( parent.getNode (  )  positionnstanceof Element )  {^45^^^^^30^60^if  ( parent.getNode (  )  instanceof Element )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( parent.getNode (  )   &  Element )  {^45^^^^^30^60^if  ( parent.getNode (  )  instanceof Element )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^if  ( lname != null )  {^49^^^^^34^64^if  ( prefix != null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( prefix == null )  {^49^^^^^34^64^if  ( prefix != null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^if  ( lname.equals ( "xml" )  )  {^50^^^^^35^65^if  ( prefix.equals ( "xml" )  )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( ns != null )  {^55^^^^^50^60^if  ( ns == null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^ns = element.getNamespace ( lname ) ;^54^^^^^50^60^ns = element.getNamespace ( prefix ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^ns = prefix.getNamespace ( element ) ;^54^^^^^50^60^ns = element.getNamespace ( prefix ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( ns != null )  {^55^^^^^40^70^if  ( ns == null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^ns = element.getNamespace ( lname ) ;^54^^^^^39^69^ns = element.getNamespace ( prefix ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^ns = prefix.getNamespace ( element ) ;^54^^^^^39^69^ns = element.getNamespace ( prefix ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^if  ( ns.getNamespace (  ) .equals ( attr )  )  {^81^^^^^67^85^if  ( attr.getNamespace (  ) .equals ( ns )  )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^for  ( positionnt i = 0; i < allAttributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < attributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= allAttributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = i; i < allAttributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^Attribute attr =  ( Attribute )  attributes.get ( i ) ;^80^^^^^67^85^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^Attribute attr =  ( Attribute )  allAttributes.get ( position ) ;^80^^^^^67^85^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^Attribute attr =  ( Attribute )  i.get ( allAttributes ) ;^80^^^^^67^85^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = position; i < allAttributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( ns == null )  {^69^^^^^54^84^if  ( ns != null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^if  ( attr == null )  {^71^^^^^56^86^if  ( attr != null )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^Attribute attr = ns.getAttribute ( lname, element ) ;^70^^^^^55^85^Attribute attr = element.getAttribute ( lname, ns ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^Attribute attr = lname.getAttribute ( element, ns ) ;^70^^^^^55^85^Attribute attr = element.getAttribute ( lname, ns ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^Attribute attr = element.getAttribute ( prefix, ns ) ;^70^^^^^55^85^Attribute attr = element.getAttribute ( lname, ns ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^Attribute attr = element.getAttribute ( ns, lname ) ;^70^^^^^55^85^Attribute attr = element.getAttribute ( lname, ns ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^if  ( ns.getNamespace (  ) .equals ( attr )  )  {^81^^^^^66^96^if  ( attr.getNamespace (  ) .equals ( ns )  )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^for  ( positionnt i = 0; i < allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < attributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^Attribute attr =  ( Attribute )  allAttributes.get ( position ) ;^80^^^^^65^95^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Argument_Swapping]^Attribute attr =  ( Attribute )  i.get ( allAttributes ) ;^80^^^^^65^95^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^Attribute attr =  ( Attribute )  attributes.get ( i ) ;^80^^^^^65^95^Attribute attr =  ( Attribute )  allAttributes.get ( i ) ;^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = ; i < allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = ; i < allAttributes.size (  ) ; i++ )  {^79^^^^^67^85^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = position; i < allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < allAttributes.size (  ) ; i++ )  {^79^^^^^64^94^for  ( int i = 0; i < allAttributes.size (  ) ; i++ )  {^[CLASS] JDOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] boolean  Attribute  attr  QName  name  Element  element  List  allAttributes  attributes  NodePointer  parent  String  lname  prefix  int  i  position  Namespace  ns  
[BugLab_Variable_Misuse]^if  ( index == 0 )  {^164^^^^^163^177^if  ( position == 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^if  ( position >= 0 )  {^164^^^^^163^177^if  ( position == 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^if  ( position == -1 )  {^164^^^^^163^177^if  ( position == 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^position = index;^168^^^^^163^177^position = 0;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^if  ( !setPosition ( 2 )  )  {^165^^^^^163^177^if  ( !setPosition ( 1 )  )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^if  ( !setPosition (  )  )  {^165^^^^^163^177^if  ( !setPosition ( 1 )  )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^position = 1;^168^^^^^163^177^position = 0;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^int index = i - 1;^170^^^^^163^177^int index = position - 1;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^int index = position  ==  1;^170^^^^^163^177^int index = position - 1;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^int index = position - i;^170^^^^^163^177^int index = position - 1;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^if  ( position < 0 )  {^171^^^^^163^177^if  ( index < 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^if  ( index == 0 )  {^171^^^^^163^177^if  ( index < 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^if  ( index < 1 )  {^171^^^^^163^177^if  ( index < 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^if  ( index <  )  {^171^^^^^163^177^if  ( index < 0 )  {^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^index = index;^172^^^^^163^177^index = 0;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^index = position;^172^^^^^163^177^index = 0;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( position )  ) ;^174^175^176^^^163^177^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^return new JDOMAttributePointer ( parent, ( Attribute )  allAttributes.get ( index )  ) ;^174^175^176^^^163^177^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Argument_Swapping]^return new JDOMAttributePointer ( parent, ( Attribute )  index.get ( attributes )  ) ;^174^175^176^^^163^177^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Argument_Swapping]^return new JDOMAttributePointer ( attributes, ( Attribute )  parent.get ( index )  ) ;^174^175^176^^^163^177^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Argument_Swapping]^return new JDOMAttributePointer ( index, ( Attribute )  attributes.get ( parent )  ) ;^174^175^176^^^163^177^return new JDOMAttributePointer ( parent, ( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^( Attribute )  attributes.get ( position )  ) ;^176^^^^^163^177^( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Argument_Swapping]^( Attribute )  index.get ( attributes )  ) ;^176^^^^^163^177^( Attribute )  attributes.get ( index )  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^return index;^180^^^^^179^181^return position;^[CLASS] JDOMAttributeIterator  [METHOD] getPosition [RETURN_TYPE] int   [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^if  ( allAttributes == null )  {^184^^^^^183^189^if  ( attributes == null )  {^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^if  ( attributes != null )  {^184^^^^^183^189^if  ( attributes == null )  {^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^return true;^185^^^^^183^189^return false;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^this.position = index;^187^^^^^183^189^this.position = position;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^return index >= 1 && position <= attributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Variable_Misuse]^return position >= 1 && position <= allAttributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Argument_Swapping]^return attributes >= 1 && position <= position.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^return position >= 1 || position <= attributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^return position > 1 && position <= attributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Operator]^return position >= 1 && position < attributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
[BugLab_Wrong_Literal]^return position >= index && position <= attributes.size (  ) ;^188^^^^^183^189^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] JDOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  allAttributes  attributes  NodePointer  parent  boolean  QName  name  int  i  index  position  
