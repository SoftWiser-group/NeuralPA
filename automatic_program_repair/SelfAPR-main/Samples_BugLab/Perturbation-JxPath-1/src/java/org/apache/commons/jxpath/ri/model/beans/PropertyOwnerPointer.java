[BugLab_Argument_Swapping]^super ( locale, parent ) ;^85^^^^^84^86^super ( parent, locale ) ;^[CLASS] PropertyOwnerPointer  [METHOD] <init> [RETURN_TYPE] Locale)   NodePointer parent Locale locale [VARIABLES] Locale  locale  Object  UNINITIALIZED  value  NodePointer  parent  boolean  
[BugLab_Argument_Swapping]^return createNodeIterator ( null, startWith, reverse ) ;^66^^^^^64^67^return createNodeIterator ( null, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return createNodeIterator ( null, startWith, reverse ) ;^66^^^^^44^70^return createNodeIterator ( null, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return createNodeIterator ( startWith, reverse, property ) ;^61^^^^^44^70^return createNodeIterator ( property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return createNodeIterator ( reverse, property, startWith ) ;^61^^^^^44^70^return createNodeIterator ( property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return createNodeIterator ( property, startWith, reverse ) ;^61^^^^^44^70^return createNodeIterator ( property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return createNodeIterator ( null, startWith, reverse ) ;^46^^^^^44^70^return createNodeIterator ( null, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] childIterator [RETURN_TYPE] NodeIterator   NodeTest test boolean reverse NodePointer startWith [VARIABLES] boolean  reverse  NodeNameTest  nodeNameTest  QName  testName  Object  UNINITIALIZED  value  NodePointer  startWith  String  property  NodeTest  test  
[BugLab_Argument_Swapping]^return new PropertyIterator ( this, startWith, reverse, property ) ;^77^^^^^72^78^return new PropertyIterator ( this, property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] createNodeIterator [RETURN_TYPE] NodeIterator   String property boolean reverse NodePointer startWith [VARIABLES] Object  UNINITIALIZED  value  String  property  boolean  reverse  NodePointer  startWith  
[BugLab_Argument_Swapping]^return new PropertyIterator ( this, property, startWith, reverse ) ;^77^^^^^72^78^return new PropertyIterator ( this, property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] createNodeIterator [RETURN_TYPE] NodeIterator   String property boolean reverse NodePointer startWith [VARIABLES] Object  UNINITIALIZED  value  String  property  boolean  reverse  NodePointer  startWith  
[BugLab_Argument_Swapping]^return new PropertyIterator ( this, reverse, property, startWith ) ;^77^^^^^72^78^return new PropertyIterator ( this, property, reverse, startWith ) ;^[CLASS] PropertyOwnerPointer  [METHOD] createNodeIterator [RETURN_TYPE] NodeIterator   String property boolean reverse NodePointer startWith [VARIABLES] Object  UNINITIALIZED  value  String  property  boolean  reverse  NodePointer  startWith  
[BugLab_Wrong_Operator]^if  ( this.index == index )  {^93^^^^^92^97^if  ( this.index != index )  {^[CLASS] PropertyOwnerPointer  [METHOD] setIndex [RETURN_TYPE] void   int index [VARIABLES] Object  UNINITIALIZED  value  int  index  boolean  
[BugLab_Variable_Misuse]^value = value;^95^^^^^92^97^value = UNINITIALIZED;^[CLASS] PropertyOwnerPointer  [METHOD] setIndex [RETURN_TYPE] void   int index [VARIABLES] Object  UNINITIALIZED  value  int  index  boolean  
[BugLab_Argument_Swapping]^if  ( UNINITIALIZED == value )  {^103^^^^^102^112^if  ( value == UNINITIALIZED )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( value >= UNINITIALIZED )  {^103^^^^^102^112^if  ( value == UNINITIALIZED )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( value <= UNINITIALIZED )  {^103^^^^^102^112^if  ( value == UNINITIALIZED )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Variable_Misuse]^if  ( index == null )  {^104^^^^^102^112^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Argument_Swapping]^if  ( WHOLE_COLLECTION == index )  {^104^^^^^102^112^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( index <= WHOLE_COLLECTION )  {^104^^^^^102^112^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Variable_Misuse]^return UNINITIALIZED;^111^^^^^102^112^return value;^[CLASS] PropertyOwnerPointer  [METHOD] getImmediateNode [RETURN_TYPE] Object   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Variable_Misuse]^this.value = UNINITIALIZED;^121^^^^^120^140^this.value = value;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^else if  ( parent == null )  {^125^^^^^120^140^else if  ( parent != null )  {^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Argument_Swapping]^if  ( WHOLE_COLLECTION == index )  {^126^^^^^120^140^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( index != WHOLE_COLLECTION )  {^126^^^^^120^140^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  ||  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  >=  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  ==  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  ^  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  >=  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  >  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  <<  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( index <= WHOLE_COLLECTION )  {^126^^^^^120^140^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  >>  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  <=  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  |  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  ^  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Variable_Misuse]^parent.setValue ( UNINITIALIZED ) ;^123^^^^^120^140^parent.setValue ( value ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( index < WHOLE_COLLECTION )  {^126^^^^^120^140^if  ( index == WHOLE_COLLECTION )  {^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  ==  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: "  !=  this ) ;^132^133^^^^120^140^throw new JXPathInvalidAccessException ( "The specified collection element does not exist: " + this ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  >  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  &&  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "   instanceof   "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  &  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not "  <=  "some other object's property" ) ;^127^128^129^^^120^140^throw new UnsupportedOperationException ( "Cannot setValue of an object that is not " + "some other object's property" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] setValue [RETURN_TYPE] void   Object value [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^if  ( parent == null )  {^148^^^^^146^156^if  ( parent != null )  {^[CLASS] PropertyOwnerPointer  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot remove an object that is not "  ||  "some other object's property or a collection element" ) ;^152^153^154^^^146^156^throw new UnsupportedOperationException ( "Cannot remove an object that is not " + "some other object's property or a collection element" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot remove an object that is not "  <<  "some other object's property or a collection element" ) ;^152^153^154^^^146^156^throw new UnsupportedOperationException ( "Cannot remove an object that is not " + "some other object's property or a collection element" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot remove an object that is not "  <  "some other object's property or a collection element" ) ;^152^153^154^^^146^156^throw new UnsupportedOperationException ( "Cannot remove an object that is not " + "some other object's property or a collection element" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Operator]^throw new UnsupportedOperationException ( "Cannot remove an object that is not "  ==  "some other object's property or a collection element" ) ;^152^153^154^^^146^156^throw new UnsupportedOperationException ( "Cannot remove an object that is not " + "some other object's property or a collection element" ) ;^[CLASS] PropertyOwnerPointer  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Wrong_Literal]^return true;^166^^^^^165^167^return false;^[CLASS] PropertyOwnerPointer  [METHOD] isDynamicPropertyDeclarationSupported [RETURN_TYPE] boolean   [VARIABLES] Object  UNINITIALIZED  value  boolean  
[BugLab_Variable_Misuse]^int r = pointer2.getName (  ) .toString (  ) .compareTo ( pointer2.getName (  ) .toString (  )  ) ;^173^174^175^^^169^180^int r = pointer1.getName (  ) .toString (  ) .compareTo ( pointer2.getName (  ) .toString (  )  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Variable_Misuse]^int r = pointer1.getName (  ) .toString (  ) .compareTo ( pointer1.getName (  ) .toString (  )  ) ;^173^174^175^^^169^180^int r = pointer1.getName (  ) .toString (  ) .compareTo ( pointer2.getName (  ) .toString (  )  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Argument_Swapping]^int r = pointer2.getName (  ) .toString (  ) .compareTo ( pointer1.getName (  ) .toString (  )  ) ;^173^174^175^^^169^180^int r = pointer1.getName (  ) .toString (  ) .compareTo ( pointer2.getName (  ) .toString (  )  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Wrong_Operator]^if  ( r >= 0 )  {^176^^^^^169^180^if  ( r != 0 )  {^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Argument_Swapping]^return pointer2.getIndex (  )  - pointer1.getIndex (  ) ;^179^^^^^169^180^return pointer1.getIndex (  )  - pointer2.getIndex (  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Wrong_Operator]^return pointer1.getIndex (  )   &  pointer2.getIndex (  ) ;^179^^^^^169^180^return pointer1.getIndex (  )  - pointer2.getIndex (  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Variable_Misuse]^return pointer2.getIndex (  )  - pointer2.getIndex (  ) ;^179^^^^^169^180^return pointer1.getIndex (  )  - pointer2.getIndex (  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
[BugLab_Variable_Misuse]^return pointer1.getIndex (  )  - pointer1.getIndex (  ) ;^179^^^^^169^180^return pointer1.getIndex (  )  - pointer2.getIndex (  ) ;^[CLASS] PropertyOwnerPointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Object  UNINITIALIZED  value  NodePointer  pointer1  pointer2  boolean  int  r  
