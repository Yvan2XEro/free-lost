import { View, Text, Image, Animated } from 'react-native';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import Icon from 'react-native-vector-icons/FontAwesome';
import { TouchableOpacity } from 'react-native';
import { Button } from 'react-native-elements';
// import { Interpreter } from '@tensorflow/tfjs-react-native';
import * as tf from '@tensorflow/tfjs';
// import { loadGraphModel } from '@tensorflow/tfjs-converter';


export default function ImageSearch({ navigation }) {

    useEffect(() => {
        const loadModel = async () => {
            try {
                await tf.ready();
                const modelJson = ('../assets/rn_model/model.json');
                const modelWeights = ('../assets/rn_model/group1-shard.bin');
                const model = await tf.loadLayersModel(
                    modelJson,
                    modelWeights
                );
                console.log(model)
            } catch (err) {
                console.log('Erreur lors du chargement du modèle :', err);
            }
        };

        loadModel()
    }, [])

    const [img, setImg] = useState();
    const [loading, setloading] = useState(false);


    const pickImage = useCallback(() => {
        launchImageLibrary({ mediaType: 'photo' }).then((result) => {
            if (!result.cancelled) {
                setImg(result.assets[0].uri);
            }
        });
    }, []);

    const pickWithCamera = useCallback(() => {
        launchCamera({ mediaType: 'photo' }).then((result) => {
            if (!result.cancelled) {
                setImg(result.assets[0].uri);
            }
        });
    }, []);
    const opacity = useRef(new Animated.Value(0)).current;

    const fadeInOut = () => {
        Animated.sequence([
            Animated.timing(opacity, {
                toValue: 1,
                duration: 200,
                useNativeDriver: true,
            }),
            Animated.timing(opacity, {
                toValue: 0,
                duration: 200,
                delay: 1800,
                useNativeDriver: true,
            }),
        ]).start(() => fadeInOut());
    };
    useEffect(() => {
        fadeInOut();
    }, []);

    async function wait() {
        await new Promise((resolve) => setTimeout(resolve, 4000));
    }

    async function onSearch() {

        console.log(img);
        setloading(true)
        await wait()
        await getPredictions(img)
        setloading(false)
        navigation.navigate("Details d'un Object", {
            otherData: { ...routeParams, imgP: img },
        })
    }

    return (
        <View style={{ display: 'flex', alignItems: 'center', height: '100%', justifyContent: 'space-between', padding: 20 }}>
            <View style={{ display: 'flex', justifyContent: 'space-evenly', flexDirection: 'row', width: '100%' }}>
                <TouchableOpacity onPress={pickWithCamera} style={{ display: 'flex', alignItems: 'center', padding: 20, borderWidth: 1, borderRadius: 10, flex: 0.4 }}>
                    <Icon name="camera" size={30} />
                    <Text>Appareil photo</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={pickImage} style={{ display: 'flex', alignItems: 'center', padding: 20, borderWidth: 1, borderRadius: 10, flex: 0.4 }}>
                    <Icon name="image" size={30} />
                    <Text>Selectionner une photo</Text>
                </TouchableOpacity>
            </View>
            {!!img && (
                <View style={{ position: 'relative', overflow: "hidden" }}>
                    <Image source={{ uri: img }} style={{ width: 300, height: 300 }} />
                    {loading && <>
                        {Array.from({ length: 70 }).map((_, i) => (
                            <Animated.View
                                style={{
                                    position: 'absolute',
                                    top: Math.random() * 300, // Modifier les valeurs selon vos besoins
                                    left: Math.random() * 300, // Modifier les valeurs selon vos besoins
                                    opacity,
                                    width: 5,
                                    height: 5,
                                    backgroundColor: 'yellow',
                                    borderRadius: 2.5,
                                }}
                            />
                        ))}</>}
                </View>
            )}
            <Button
                onPress={onSearch}
                buttonStyle={{
                    backgroundColor: 'green',
                    width: 200,
                    marginTop: 8,
                }}
                disabled={!img}
                titleStyle={{
                    textAlign: 'center',
                    fontWeight: 'bold',
                    fontSize: 25,
                }}
                title="Search"
            />
        </View>
    );
}

const transformImageToTensor = async (uri) => {

    const img64 = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 })
    const imgBuffer = tf.util.encodeString(img64, 'base64').buffer
    const raw = new Uint8Array(imgBuffer)
    let imgTensor = decodeJpeg(raw)
    const scalar = tf.scalar(255)
    imgTensor = tf.image.resizeNearestNeighbor(imgTensor, [200, 200])
    const tensorScaled = imgTensor.div(scalar)
    const img = tf.reshape(tensorScaled, [1, 200, 200, 3])
    return img
}

const getPredictions = async (image) => {
    await tf.ready()
    tf.setPlatform('react-native');
    const modelJson = ('../assets/rn_model/model.json');
    const modelWeights = ('../assets/rn_model/group1-shard.bin');
    const model = await tf.loadLayersModel(
        modelJson,
        (modelWeights)
    );
    const tensor_image = await transformImageToTensor(image)
    const predictions = await makePredictions(1, model, tensor_image)
    return predictions
}
const makePredictions = async (batch, model, imagesTensor) => {
    const predictionsdata = model.predict(imagesTensor)
    let pred = predictionsdata.split(batch)
    return pred
}





















const routeParams = { "Categorie": "Papiers et documents officiels", "DatePerteObjet": "Thu Jul 14 2022", "Description": "Carte retrouvé au campus de l'UDS", "Lieu": "DSCHANG ", "Marque": "", "Modele": "", "Nationalite": "CAMEROUNAISE ", "NomDoc": "Carte CNI", "Photo": "/images/objects/1657801934099-rn_image_picker_lib_temp_86231b7b-d276-4ae9-ac06-11ada99f9d36.jpg", "UserNom": "Kana", "UserPrenom": "Jaures", "__v": 0, "_id": "62d00ccff19754be940d2658", "idModerateur": "62fa342cde0935b1ed1ebc3d", "idUtilisateur": "62ccad588f1b65f689851280", "nomObjet": "", "valider": "oui" }