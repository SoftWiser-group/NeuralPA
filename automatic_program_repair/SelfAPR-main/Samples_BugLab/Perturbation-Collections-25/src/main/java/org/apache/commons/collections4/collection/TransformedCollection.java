[BugLab_Wrong_Operator]^if  ( transformer != null )  {^110^^^^^108^114^if  ( transformer == null )  {^[CLASS] TransformedCollection  [METHOD] <init> [RETURN_TYPE] Transformer)   Collection<E> coll Transformer<? super E, ? extends E> transformer [VARIABLES] Collection  coll  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new TransformedCollection<E> ( transformer, coll ) ;^63^^^^^61^64^return new TransformedCollection<E> ( coll, transformer ) ;^[CLASS] TransformedCollection  [METHOD] transformingCollection [RETURN_TYPE] <E>   Collection<E> coll Transformer<? super E, ? extends E> transformer [VARIABLES] Collection  coll  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new TransformedCollection<E> ( coll, 1 ) ;^63^^^^^61^64^return new TransformedCollection<E> ( coll, transformer ) ;^[CLASS] TransformedCollection  [METHOD] transformingCollection [RETURN_TYPE] <E>   Collection<E> coll Transformer<? super E, ? extends E> transformer [VARIABLES] Collection  coll  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^final TransformedCollection<E> decorated = new TransformedCollection<E> ( transformer, collection ) ;^84^^^^^81^95^final TransformedCollection<E> decorated = new TransformedCollection<E> ( collection, transformer ) ;^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( collection.size (  )  >= 0 )  {^86^^^^^81^95^if  ( collection.size (  )  > 0 )  {^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( collection.size (  )  > -1 )  {^86^^^^^81^95^if  ( collection.size (  )  > 0 )  {^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Variable_Misuse]^decorated.decorated (  ) .add ( null.transform ( value )  ) ;^91^^^^^81^95^decorated.decorated (  ) .add ( transformer.transform ( value )  ) ;^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Argument_Swapping]^decorated.decorated (  ) .add ( value.transform ( transformer )  ) ;^91^^^^^81^95^decorated.decorated (  ) .add ( transformer.transform ( value )  ) ;^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Variable_Misuse]^return null;^94^^^^^81^95^return decorated;^[CLASS] TransformedCollection  [METHOD] transformedCollection [RETURN_TYPE] <E>   Collection<E> collection Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  TransformedCollection  decorated  Collection  collection  long  serialVersionUID  
[BugLab_Argument_Swapping]^return object.transform ( transformer ) ;^125^^^^^124^126^return transformer.transform ( object ) ;^[CLASS] TransformedCollection  [METHOD] transform [RETURN_TYPE] E   final E object [VARIABLES] Transformer  transformer  boolean  E  object  long  serialVersionUID  
[BugLab_Variable_Misuse]^return decorated (  ) .addAll ( transform ( 1 )  ) ;^152^^^^^151^153^return decorated (  ) .addAll ( transform ( coll )  ) ;^[CLASS] TransformedCollection  [METHOD] addAll [RETURN_TYPE] boolean   Collection<? extends E> coll [VARIABLES] Collection  coll  Transformer  transformer  boolean  long  serialVersionUID  
